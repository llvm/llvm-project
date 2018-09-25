//===-- hwasan_report.cc --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// Error reporting.
//===----------------------------------------------------------------------===//

#include "hwasan.h"
#include "hwasan_allocator.h"
#include "hwasan_mapping.h"
#include "hwasan_thread.h"
#include "hwasan_thread_list.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __sanitizer;

namespace __hwasan {

static StackTrace GetStackTraceFromId(u32 id) {
  CHECK(id);
  StackTrace res = StackDepotGet(id);
  CHECK(res.trace);
  return res;
}

// A RAII object that holds a copy of the current thread stack ring buffer.
// The actual stack buffer may change while we are iterating over it (for
// example, Printf may call syslog() which can itself be built with hwasan).
class SavedStackAllocations {
 public:
  SavedStackAllocations(StackAllocationsRingBuffer *rb) {
    uptr size = rb->size() * sizeof(uptr);
    void *storage =
        MmapAlignedOrDieOnFatalError(size, size * 2, "saved stack allocations");
    new (&rb_) StackAllocationsRingBuffer(*rb, storage);
  }

  ~SavedStackAllocations() {
    StackAllocationsRingBuffer *rb = get();
    UnmapOrDie(rb->StartOfStorage(), rb->size() * sizeof(uptr));
  }

  StackAllocationsRingBuffer *get() {
    return (StackAllocationsRingBuffer *)&rb_;
  }

 private:
  uptr rb_;
};

class Decorator: public __sanitizer::SanitizerCommonDecorator {
 public:
  Decorator() : SanitizerCommonDecorator() { }
  const char *Access() { return Blue(); }
  const char *Allocation() const { return Magenta(); }
  const char *Origin() const { return Magenta(); }
  const char *Name() const { return Green(); }
  const char *Location() { return Green(); }
  const char *Thread() { return Green(); }
};

// Returns the index of the rb element that matches tagged_addr (plus one),
// or zero if found nothing.
uptr FindHeapAllocation(HeapAllocationsRingBuffer *rb,
                        uptr tagged_addr,
                        HeapAllocationRecord *har) {
  if (!rb) return 0;
  for (uptr i = 0, size = rb->size(); i < size; i++) {
    auto h = (*rb)[i];
    if (h.tagged_addr <= tagged_addr &&
        h.tagged_addr + h.requested_size > tagged_addr) {
      *har = h;
      return i + 1;
    }
  }
  return 0;
}

void PrintAddressDescription(
    uptr tagged_addr, uptr access_size,
    StackAllocationsRingBuffer *current_stack_allocations) {
  Decorator d;
  int num_descriptions_printed = 0;
  uptr untagged_addr = UntagAddr(tagged_addr);
  // Check if this looks like a heap buffer overflow by scanning
  // the shadow left and right and looking for the first adjacent
  // object with a different memory tag. If that tag matches addr_tag,
  // check the allocator if it has a live chunk there.
  tag_t addr_tag = GetTagFromPointer(tagged_addr);
  tag_t *tag_ptr = reinterpret_cast<tag_t*>(MemToShadow(untagged_addr));
  if (*tag_ptr != addr_tag) { // should be true usually.
    tag_t *left = tag_ptr, *right = tag_ptr;
    // scan left.
    for (int i = 0; i < 1000 && *left == *tag_ptr; i++, left--){}
    // scan right.
    for (int i = 0; i < 1000 && *right == *tag_ptr; i++, right++){}
    // Chose the object that has addr_tag and that is closer to addr.
    tag_t *candidate = nullptr;
    if (*right == addr_tag && *left == addr_tag)
      candidate = right - tag_ptr < tag_ptr - left ? right : left;
    else if (*right == addr_tag)
      candidate = right;
    else if (*left == addr_tag)
      candidate = left;

    if (candidate) {
      uptr mem = ShadowToMem(reinterpret_cast<uptr>(candidate));
      HwasanChunkView chunk = FindHeapChunkByAddress(mem);
      if (chunk.IsAllocated()) {
        Printf("%s", d.Location());
        Printf(
            "%p is located %zd bytes to the %s of %zd-byte region [%p,%p)\n",
            untagged_addr,
            candidate == left ? untagged_addr - chunk.End()
            : chunk.Beg() - untagged_addr,
            candidate == right ? "left" : "right", chunk.UsedSize(),
            chunk.Beg(), chunk.End());
        Printf("%s", d.Allocation());
        Printf("allocated here:\n");
        Printf("%s", d.Default());
        GetStackTraceFromId(chunk.GetAllocStackId()).Print();
        num_descriptions_printed++;
      }
    }
  }

  hwasanThreadList().VisitAllLiveThreads([&](Thread *t) {
    // Scan all threads' ring buffers to find if it's a heap-use-after-free.
    HeapAllocationRecord har;
    if (uptr D = FindHeapAllocation(t->heap_allocations(), tagged_addr, &har)) {
      Printf("%s", d.Location());
      Printf("%p is located %zd bytes inside of %zd-byte region [%p,%p)\n",
             untagged_addr, untagged_addr - UntagAddr(har.tagged_addr),
             har.requested_size, UntagAddr(har.tagged_addr),
             UntagAddr(har.tagged_addr) + har.requested_size);
      Printf("%s", d.Allocation());
      Printf("freed by thread T%zd here:\n", t->unique_id());
      Printf("%s", d.Default());
      GetStackTraceFromId(har.free_context_id).Print();

      Printf("%s", d.Allocation());
      Printf("previously allocated here:\n", t);
      Printf("%s", d.Default());
      GetStackTraceFromId(har.alloc_context_id).Print();
      t->Announce();

      // Print a developer note: the index of this heap object
      // in the thread's deallocation ring buffer.
      Printf("hwasan_dev_note_heap_rb_distance: %zd %zd\n", D,
             flags()->heap_history_size);

      num_descriptions_printed++;
    }

    // Very basic check for stack memory.
    if (t->AddrIsInStack(untagged_addr)) {
      Printf("%s", d.Location());
      Printf("Address %p is located in stack of thread T%zd\n", untagged_addr,
             t->unique_id());
      Printf("%s", d.Default());
      t->Announce();

      // Temporary report section, needs to be improved.
      Printf("Previosly allocated frames:\n");
      auto *sa = (t == GetCurrentThread() && current_stack_allocations)
                     ? current_stack_allocations
                     : t->stack_allocations();
      uptr frames = Min((uptr)flags()->stack_history_size, sa->size());
      for (uptr i = 0; i < frames; i++) {
        uptr record = (*sa)[i];
        if (!record)
          break;
        uptr sp = (record >> 48) << 4;
        uptr pc_mask = (1ULL << 48) - 1;
        uptr pc = record & pc_mask;
        uptr fixed_pc = StackTrace::GetNextInstructionPc(pc);
        StackTrace stack(&fixed_pc, 1);
        Printf("record: %p pc: %p sp: %p", record, pc, sp);
        stack.Print();
      }

      num_descriptions_printed++;
    }
  });

  if (!num_descriptions_printed)
    // We exhausted our possibilities. Bail out.
    Printf("HWAddressSanitizer can not describe address in more detail.\n");
}

void ReportInvalidAccess(StackTrace *stack, u32 origin) {
  ScopedErrorReportLock l;

  Decorator d;
  Printf("%s", d.Warning());
  Report("WARNING: HWAddressSanitizer: invalid access\n");
  Printf("%s", d.Default());
  stack->Print();
  ReportErrorSummary("invalid-access", stack);
}

void ReportStats() {}

void ReportInvalidAccessInsideAddressRange(const char *what, const void *start,
                                           uptr size, uptr offset) {
  ScopedErrorReportLock l;
  SavedStackAllocations current_stack_allocations(
      GetCurrentThread()->stack_allocations());

  Decorator d;
  Printf("%s", d.Warning());
  Printf("%sTag mismatch in %s%s%s at offset %zu inside [%p, %zu)%s\n",
         d.Warning(), d.Name(), what, d.Warning(), offset, start, size,
         d.Default());
  PrintAddressDescription((uptr)start + offset, 1,
                          current_stack_allocations.get());
  // if (__sanitizer::Verbosity())
  //   DescribeMemoryRange(start, size);
}

static void PrintTagsAroundAddr(tag_t *tag_ptr) {
  Printf(
      "Memory tags around the buggy address (one tag corresponds to %zd "
      "bytes):\n", kShadowAlignment);

  const uptr row_len = 16;  // better be power of two.
  const uptr num_rows = 11;
  tag_t *center_row_beg = reinterpret_cast<tag_t *>(
      RoundDownTo(reinterpret_cast<uptr>(tag_ptr), row_len));
  tag_t *beg_row = center_row_beg - row_len * (num_rows / 2);
  tag_t *end_row = center_row_beg + row_len * (num_rows / 2);
  InternalScopedString s(GetPageSizeCached());
  for (tag_t *row = beg_row; row < end_row; row += row_len) {
    s.append("%s", row == center_row_beg ? "=>" : "  ");
    for (uptr i = 0; i < row_len; i++) {
      s.append("%s", row + i == tag_ptr ? "[" : " ");
      s.append("%02x", row[i]);
      s.append("%s", row + i == tag_ptr ? "]" : " ");
    }
    s.append("%s\n", row == center_row_beg ? "<=" : "  ");
    Printf("%s", s.data());
    s.clear();
  }
}

void ReportInvalidFree(StackTrace *stack, uptr tagged_addr) {
  ScopedErrorReportLock l;
  uptr untagged_addr = UntagAddr(tagged_addr);
  tag_t ptr_tag = GetTagFromPointer(tagged_addr);
  tag_t *tag_ptr = reinterpret_cast<tag_t*>(MemToShadow(untagged_addr));
  tag_t mem_tag = *tag_ptr;
  Decorator d;
  Printf("%s", d.Error());
  uptr pc = stack->size ? stack->trace[0] : 0;
  const char *bug_type = "invalid-free";
  Report("ERROR: %s: %s on address %p at pc %p\n", SanitizerToolName, bug_type,
         untagged_addr, pc);
  Printf("%s", d.Access());
  Printf("tags: %02x/%02x (ptr/mem)\n", ptr_tag, mem_tag);
  Printf("%s", d.Default());

  stack->Print();

  PrintAddressDescription(tagged_addr, 0, nullptr);

  PrintTagsAroundAddr(tag_ptr);

  ReportErrorSummary(bug_type, stack);
  Die();
}

void ReportTagMismatch(StackTrace *stack, uptr tagged_addr, uptr access_size,
                       bool is_store) {
  ScopedErrorReportLock l;
  SavedStackAllocations current_stack_allocations(
      GetCurrentThread()->stack_allocations());

  Decorator d;
  Printf("%s", d.Error());
  uptr untagged_addr = UntagAddr(tagged_addr);
  // TODO: when possible, try to print heap-use-after-free, etc.
  const char *bug_type = "tag-mismatch";
  uptr pc = stack->size ? stack->trace[0] : 0;
  Report("ERROR: %s: %s on address %p at pc %p\n", SanitizerToolName, bug_type,
         untagged_addr, pc);

  Thread *t = GetCurrentThread();

  tag_t ptr_tag = GetTagFromPointer(tagged_addr);
  tag_t *tag_ptr = reinterpret_cast<tag_t*>(MemToShadow(untagged_addr));
  tag_t mem_tag = *tag_ptr;
  Printf("%s", d.Access());
  Printf("%s of size %zu at %p tags: %02x/%02x (ptr/mem) in thread T%zd\n",
         is_store ? "WRITE" : "READ", access_size, untagged_addr, ptr_tag,
         mem_tag, t->unique_id());
  Printf("%s", d.Default());

  stack->Print();

  PrintAddressDescription(tagged_addr, access_size,
                          current_stack_allocations.get());
  t->Announce();

  PrintTagsAroundAddr(tag_ptr);

  ReportErrorSummary(bug_type, stack);
}

}  // namespace __hwasan
