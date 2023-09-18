//===-- hwasan_report.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// Error reporting.
//===----------------------------------------------------------------------===//

#include "hwasan_report.h"

#include <dlfcn.h>

#include "hwasan.h"
#include "hwasan_allocator.h"
#include "hwasan_globals.h"
#include "hwasan_mapping.h"
#include "hwasan_thread.h"
#include "hwasan_thread_list.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_array_ref.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace_printer.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __sanitizer;

namespace __hwasan {

class ScopedReport {
 public:
  explicit ScopedReport(bool fatal) : fatal(fatal) {
    Lock lock(&error_message_lock_);
    error_message_ptr_ = fatal ? &error_message_ : nullptr;
    ++hwasan_report_count;
  }

  ~ScopedReport() {
    void (*report_cb)(const char *);
    {
      Lock lock(&error_message_lock_);
      report_cb = error_report_callback_;
      error_message_ptr_ = nullptr;
    }
    if (report_cb)
      report_cb(error_message_.data());
    if (fatal)
      SetAbortMessage(error_message_.data());
    if (common_flags()->print_module_map >= 2 ||
        (fatal && common_flags()->print_module_map))
      DumpProcessMap();
    if (fatal)
      Die();
  }

  static void MaybeAppendToErrorMessage(const char *msg) {
    Lock lock(&error_message_lock_);
    if (!error_message_ptr_)
      return;
    error_message_ptr_->Append(msg);
  }

  static void SetErrorReportCallback(void (*callback)(const char *)) {
    Lock lock(&error_message_lock_);
    error_report_callback_ = callback;
  }

 private:
  InternalScopedString error_message_;
  bool fatal;

  static Mutex error_message_lock_;
  static InternalScopedString *error_message_ptr_
      SANITIZER_GUARDED_BY(error_message_lock_);
  static void (*error_report_callback_)(const char *);
};

Mutex ScopedReport::error_message_lock_;
InternalScopedString *ScopedReport::error_message_ptr_;
void (*ScopedReport::error_report_callback_)(const char *);

// If there is an active ScopedReport, append to its error message.
void AppendToErrorMessageBuffer(const char *buffer) {
  ScopedReport::MaybeAppendToErrorMessage(buffer);
}

static StackTrace GetStackTraceFromId(u32 id) {
  CHECK(id);
  StackTrace res = StackDepotGet(id);
  CHECK(res.trace);
  return res;
}

static void MaybePrintAndroidHelpUrl() {
#if SANITIZER_ANDROID
  Printf(
      "Learn more about HWASan reports: "
      "https://source.android.com/docs/security/test/memory-safety/"
      "hwasan-reports\n");
#endif
}

namespace {
// A RAII object that holds a copy of the current thread stack ring buffer.
// The actual stack buffer may change while we are iterating over it (for
// example, Printf may call syslog() which can itself be built with hwasan).
class SavedStackAllocations {
 public:
  SavedStackAllocations() = default;

  explicit SavedStackAllocations(Thread *t) { CopyFrom(t); }

  void CopyFrom(Thread *t) {
    StackAllocationsRingBuffer *rb = t->stack_allocations();
    uptr size = rb->size() * sizeof(uptr);
    void *storage =
        MmapAlignedOrDieOnFatalError(size, size * 2, "saved stack allocations");
    new (&rb_) StackAllocationsRingBuffer(*rb, storage);
    thread_id_ = t->unique_id();
  }

  ~SavedStackAllocations() {
    if (rb_) {
      StackAllocationsRingBuffer *rb = get();
      UnmapOrDie(rb->StartOfStorage(), rb->size() * sizeof(uptr));
    }
  }

  const StackAllocationsRingBuffer *get() const {
    return (const StackAllocationsRingBuffer *)&rb_;
  }

  StackAllocationsRingBuffer *get() {
    return (StackAllocationsRingBuffer *)&rb_;
  }

  u32 thread_id() const { return thread_id_; }

 private:
  uptr rb_ = 0;
  u32 thread_id_;
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
}  // namespace

static bool FindHeapAllocation(HeapAllocationsRingBuffer *rb, uptr tagged_addr,
                               HeapAllocationRecord *har, uptr *ring_index,
                               uptr *num_matching_addrs,
                               uptr *num_matching_addrs_4b) {
  if (!rb) return false;

  *num_matching_addrs = 0;
  *num_matching_addrs_4b = 0;
  for (uptr i = 0, size = rb->size(); i < size; i++) {
    auto h = (*rb)[i];
    if (h.tagged_addr <= tagged_addr &&
        h.tagged_addr + h.requested_size > tagged_addr) {
      *har = h;
      *ring_index = i;
      return true;
    }

    // Measure the number of heap ring buffer entries that would have matched
    // if we had only one entry per address (e.g. if the ring buffer data was
    // stored at the address itself). This will help us tune the allocator
    // implementation for MTE.
    if (UntagAddr(h.tagged_addr) <= UntagAddr(tagged_addr) &&
        UntagAddr(h.tagged_addr) + h.requested_size > UntagAddr(tagged_addr)) {
      ++*num_matching_addrs;
    }

    // Measure the number of heap ring buffer entries that would have matched
    // if we only had 4 tag bits, which is the case for MTE.
    auto untag_4b = [](uptr p) {
      return p & ((1ULL << 60) - 1);
    };
    if (untag_4b(h.tagged_addr) <= untag_4b(tagged_addr) &&
        untag_4b(h.tagged_addr) + h.requested_size > untag_4b(tagged_addr)) {
      ++*num_matching_addrs_4b;
    }
  }
  return false;
}

static void PrintStackAllocations(const StackAllocationsRingBuffer *sa,
                                  tag_t addr_tag, uptr untagged_addr) {
  uptr frames = Min((uptr)flags()->stack_history_size, sa->size());
  bool found_local = false;
  for (uptr i = 0; i < frames; i++) {
    const uptr *record_addr = &(*sa)[i];
    uptr record = *record_addr;
    if (!record)
      break;
    tag_t base_tag =
        reinterpret_cast<uptr>(record_addr) >> kRecordAddrBaseTagShift;
    uptr fp = (record >> kRecordFPShift) << kRecordFPLShift;
    uptr pc_mask = (1ULL << kRecordFPShift) - 1;
    uptr pc = record & pc_mask;
    FrameInfo frame;
    if (Symbolizer::GetOrInit()->SymbolizeFrame(pc, &frame)) {
      for (LocalInfo &local : frame.locals) {
        if (!local.has_frame_offset || !local.has_size || !local.has_tag_offset)
          continue;
        tag_t obj_tag = base_tag ^ local.tag_offset;
        if (obj_tag != addr_tag)
          continue;
        // Calculate the offset from the object address to the faulting
        // address. Because we only store bits 4-19 of FP (bits 0-3 are
        // guaranteed to be zero), the calculation is performed mod 2^20 and may
        // harmlessly underflow if the address mod 2^20 is below the object
        // address.
        uptr obj_offset =
            (untagged_addr - fp - local.frame_offset) & (kRecordFPModulus - 1);
        if (obj_offset >= local.size)
          continue;
        if (!found_local) {
          Printf("Potentially referenced stack objects:\n");
          found_local = true;
        }
        Printf("  %s in %s %s:%d\n", local.name, local.function_name,
               local.decl_file, local.decl_line);
      }
      frame.Clear();
    }
  }

  if (found_local)
    return;

  // We didn't find any locals. Most likely we don't have symbols, so dump
  // the information that we have for offline analysis.
  InternalScopedString frame_desc;
  Printf("Previously allocated frames:\n");
  for (uptr i = 0; i < frames; i++) {
    const uptr *record_addr = &(*sa)[i];
    uptr record = *record_addr;
    if (!record)
      break;
    uptr pc_mask = (1ULL << 48) - 1;
    uptr pc = record & pc_mask;
    frame_desc.AppendF("  record_addr:0x%zx record:0x%zx",
                       reinterpret_cast<uptr>(record_addr), record);
    if (SymbolizedStack *frame = Symbolizer::GetOrInit()->SymbolizePC(pc)) {
      RenderFrame(&frame_desc, " %F %L", 0, frame->info.address, &frame->info,
                  common_flags()->symbolize_vs_style,
                  common_flags()->strip_path_prefix);
      frame->ClearAll();
    }
    Printf("%s\n", frame_desc.data());
    frame_desc.clear();
  }
}

// Returns true if tag == *tag_ptr, reading tags from short granules if
// necessary. This may return a false positive if tags 1-15 are used as a
// regular tag rather than a short granule marker.
static bool TagsEqual(tag_t tag, tag_t *tag_ptr) {
  if (tag == *tag_ptr)
    return true;
  if (*tag_ptr == 0 || *tag_ptr > kShadowAlignment - 1)
    return false;
  uptr mem = ShadowToMem(reinterpret_cast<uptr>(tag_ptr));
  tag_t inline_tag = *reinterpret_cast<tag_t *>(mem + kShadowAlignment - 1);
  return tag == inline_tag;
}

// HWASan globals store the size of the global in the descriptor. In cases where
// we don't have a binary with symbols, we can't grab the size of the global
// from the debug info - but we might be able to retrieve it from the
// descriptor. Returns zero if the lookup failed.
static uptr GetGlobalSizeFromDescriptor(uptr ptr) {
  // Find the ELF object that this global resides in.
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(ptr), &info) == 0)
    return 0;
  auto *ehdr = reinterpret_cast<const ElfW(Ehdr) *>(info.dli_fbase);
  auto *phdr_begin = reinterpret_cast<const ElfW(Phdr) *>(
      reinterpret_cast<const u8 *>(ehdr) + ehdr->e_phoff);

  // Get the load bias. This is normally the same as the dli_fbase address on
  // position-independent code, but can be different on non-PIE executables,
  // binaries using LLD's partitioning feature, or binaries compiled with a
  // linker script.
  ElfW(Addr) load_bias = 0;
  for (const auto &phdr :
       ArrayRef<const ElfW(Phdr)>(phdr_begin, phdr_begin + ehdr->e_phnum)) {
    if (phdr.p_type != PT_LOAD || phdr.p_offset != 0)
      continue;
    load_bias = reinterpret_cast<ElfW(Addr)>(ehdr) - phdr.p_vaddr;
    break;
  }

  // Walk all globals in this ELF object, looking for the one we're interested
  // in. Once we find it, we can stop iterating and return the size of the
  // global we're interested in.
  for (const hwasan_global &global :
       HwasanGlobalsFor(load_bias, phdr_begin, ehdr->e_phnum))
    if (global.addr() <= ptr && ptr < global.addr() + global.size())
      return global.size();

  return 0;
}

static void ShowHeapOrGlobalCandidate(uptr untagged_addr, tag_t *candidate,
                                      tag_t *left, tag_t *right) {
  Decorator d;
  uptr mem = ShadowToMem(reinterpret_cast<uptr>(candidate));
  HwasanChunkView chunk = FindHeapChunkByAddress(mem);
  if (chunk.IsAllocated()) {
    uptr offset;
    const char *whence;
    if (untagged_addr < chunk.End() && untagged_addr >= chunk.Beg()) {
      offset = untagged_addr - chunk.Beg();
      whence = "inside";
    } else if (candidate == left) {
      offset = untagged_addr - chunk.End();
      whence = "after";
    } else {
      offset = chunk.Beg() - untagged_addr;
      whence = "before";
    }
    Printf("%s", d.Error());
    Printf("\nCause: heap-buffer-overflow\n");
    Printf("%s", d.Default());
    Printf("%s", d.Location());
    Printf("%p is located %zd bytes %s a %zd-byte region [%p,%p)\n",
           untagged_addr, offset, whence, chunk.UsedSize(), chunk.Beg(),
           chunk.End());
    Printf("%s", d.Allocation());
    Printf("allocated by thread T%u here:\n", chunk.GetAllocThreadId());
    Printf("%s", d.Default());
    GetStackTraceFromId(chunk.GetAllocStackId()).Print();
    return;
  }
  // Check whether the address points into a loaded library. If so, this is
  // most likely a global variable.
  const char *module_name;
  uptr module_address;
  Symbolizer *sym = Symbolizer::GetOrInit();
  if (sym->GetModuleNameAndOffsetForPC(mem, &module_name, &module_address)) {
    Printf("%s", d.Error());
    Printf("\nCause: global-overflow\n");
    Printf("%s", d.Default());
    DataInfo info;
    Printf("%s", d.Location());
    if (sym->SymbolizeData(mem, &info) && info.start) {
      Printf(
          "%p is located %zd bytes %s a %zd-byte global variable "
          "%s [%p,%p) in %s\n",
          untagged_addr,
          candidate == left ? untagged_addr - (info.start + info.size)
                            : info.start - untagged_addr,
          candidate == left ? "after" : "before", info.size, info.name,
          info.start, info.start + info.size, module_name);
    } else {
      uptr size = GetGlobalSizeFromDescriptor(mem);
      if (size == 0)
        // We couldn't find the size of the global from the descriptors.
        Printf(
            "%p is located %s a global variable in "
            "\n    #0 0x%x (%s+0x%x)\n",
            untagged_addr, candidate == left ? "after" : "before", mem,
            module_name, module_address);
      else
        Printf(
            "%p is located %s a %zd-byte global variable in "
            "\n    #0 0x%x (%s+0x%x)\n",
            untagged_addr, candidate == left ? "after" : "before", size, mem,
            module_name, module_address);
    }
    Printf("%s", d.Default());
  }
}

void ReportStats() {}

static void PrintTagInfoAroundAddr(tag_t *tag_ptr, uptr num_rows,
                                   void (*print_tag)(InternalScopedString &s,
                                                     tag_t *tag)) {
  const uptr row_len = 16;  // better be power of two.
  tag_t *center_row_beg = reinterpret_cast<tag_t *>(
      RoundDownTo(reinterpret_cast<uptr>(tag_ptr), row_len));
  tag_t *beg_row = center_row_beg - row_len * (num_rows / 2);
  tag_t *end_row = center_row_beg + row_len * ((num_rows + 1) / 2);
  InternalScopedString s;
  for (tag_t *row = beg_row; row < end_row; row += row_len) {
    s.Append(row == center_row_beg ? "=>" : "  ");
    s.AppendF("%p:", (void *)ShadowToMem(reinterpret_cast<uptr>(row)));
    for (uptr i = 0; i < row_len; i++) {
      s.Append(row + i == tag_ptr ? "[" : " ");
      print_tag(s, &row[i]);
      s.Append(row + i == tag_ptr ? "]" : " ");
    }
    s.AppendF("\n");
  }
  Printf("%s", s.data());
}

static void PrintTagsAroundAddr(tag_t *tag_ptr) {
  Printf(
      "Memory tags around the buggy address (one tag corresponds to %zd "
      "bytes):\n",
      kShadowAlignment);
  PrintTagInfoAroundAddr(tag_ptr, 17, [](InternalScopedString &s, tag_t *tag) {
    s.AppendF("%02x", *tag);
  });

  Printf(
      "Tags for short granules around the buggy address (one tag corresponds "
      "to %zd bytes):\n",
      kShadowAlignment);
  PrintTagInfoAroundAddr(tag_ptr, 3, [](InternalScopedString &s, tag_t *tag) {
    if (*tag >= 1 && *tag <= kShadowAlignment) {
      uptr granule_addr = ShadowToMem(reinterpret_cast<uptr>(tag));
      s.AppendF("%02x",
                *reinterpret_cast<u8 *>(granule_addr + kShadowAlignment - 1));
    } else {
      s.AppendF("..");
    }
  });
  Printf(
      "See "
      "https://clang.llvm.org/docs/"
      "HardwareAssistedAddressSanitizerDesign.html#short-granules for a "
      "description of short granule tags\n");
}

static uptr GetTopPc(StackTrace *stack) {
  return stack->size ? StackTrace::GetPreviousInstructionPc(stack->trace[0])
                     : 0;
}

namespace {
class BaseReport {
 public:
  BaseReport(StackTrace *stack, bool fatal, uptr tagged_addr, uptr access_size)
      : scoped_report(fatal),
        stack(stack),
        tagged_addr(tagged_addr),
        access_size(access_size),
        untagged_addr(UntagAddr(tagged_addr)),
        ptr_tag(GetTagFromPointer(tagged_addr)) {
    hwasanThreadList().VisitAllLiveThreads([&](Thread *t) {
      if (stack_allocations_count < ARRAY_SIZE(stack_allocations) &&
          t->AddrIsInStack(untagged_addr)) {
        stack_allocations[stack_allocations_count++].CopyFrom(t);
      }
    });
  }

 protected:
  void PrintAddressDescription() const;

  ScopedReport scoped_report;
  StackTrace *stack = nullptr;
  uptr tagged_addr = 0;
  uptr access_size = 0;
  uptr untagged_addr = 0;
  tag_t ptr_tag = 0;
  uptr stack_allocations_count = 0;
  SavedStackAllocations stack_allocations[16];
};

void BaseReport::PrintAddressDescription() const {
  Decorator d;
  int num_descriptions_printed = 0;

  if (MemIsShadow(untagged_addr)) {
    Printf("%s%p is HWAsan shadow memory.\n%s", d.Location(), untagged_addr,
           d.Default());
    return;
  }

  // Print some very basic information about the address, if it's a heap.
  HwasanChunkView chunk = FindHeapChunkByAddress(untagged_addr);
  if (uptr beg = chunk.Beg()) {
    uptr size = chunk.ActualSize();
    Printf("%s[%p,%p) is a %s %s heap chunk; "
           "size: %zd offset: %zd\n%s",
           d.Location(),
           beg, beg + size,
           chunk.FromSmallHeap() ? "small" : "large",
           chunk.IsAllocated() ? "allocated" : "unallocated",
           size, untagged_addr - beg,
           d.Default());
  }

  // Check stack first. If the address is on the stack of a live thread, we
  // know it cannot be a heap / global overflow.
  for (uptr i = 0; i < stack_allocations_count; ++i) {
    auto &allocations = stack_allocations[i];
    // TODO(fmayer): figure out how to distinguish use-after-return and
    // stack-buffer-overflow.
    Printf("%s", d.Error());
    Printf("\nCause: stack tag-mismatch\n");
    Printf("%s", d.Location());
    Printf("Address %p is located in stack of thread T%zd\n", untagged_addr,
           allocations.thread_id());
    Printf("%s", d.Default());
    hwasanThreadList().VisitAllLiveThreads([&](Thread *t) {
      if (allocations.thread_id() == t->unique_id())
        t->Announce();
    });

    PrintStackAllocations(allocations.get(), ptr_tag, untagged_addr);
    num_descriptions_printed++;
  }

  // Check if this looks like a heap buffer overflow by scanning
  // the shadow left and right and looking for the first adjacent
  // object with a different memory tag. If that tag matches ptr_tag,
  // check the allocator if it has a live chunk there.
  tag_t *tag_ptr = reinterpret_cast<tag_t*>(MemToShadow(untagged_addr));
  tag_t *candidate = nullptr, *left = tag_ptr, *right = tag_ptr;
  uptr candidate_distance = 0;
  for (; candidate_distance < 1000; candidate_distance++) {
    if (MemIsShadow(reinterpret_cast<uptr>(left)) && TagsEqual(ptr_tag, left)) {
      candidate = left;
      break;
    }
    --left;
    if (MemIsShadow(reinterpret_cast<uptr>(right)) &&
        TagsEqual(ptr_tag, right)) {
      candidate = right;
      break;
    }
    ++right;
  }

  constexpr auto kCloseCandidateDistance = 1;

  if (!stack_allocations_count && candidate &&
      candidate_distance <= kCloseCandidateDistance) {
    ShowHeapOrGlobalCandidate(untagged_addr, candidate, left, right);
    num_descriptions_printed++;
  }

  hwasanThreadList().VisitAllLiveThreads([&](Thread *t) {
    // Scan all threads' ring buffers to find if it's a heap-use-after-free.
    HeapAllocationRecord har;
    uptr ring_index, num_matching_addrs, num_matching_addrs_4b;
    if (FindHeapAllocation(t->heap_allocations(), tagged_addr, &har,
                           &ring_index, &num_matching_addrs,
                           &num_matching_addrs_4b)) {
      Printf("%s", d.Error());
      Printf("\nCause: use-after-free\n");
      Printf("%s", d.Location());
      Printf("%p is located %zd bytes inside a %zd-byte region [%p,%p)\n",
             untagged_addr, untagged_addr - UntagAddr(har.tagged_addr),
             har.requested_size, UntagAddr(har.tagged_addr),
             UntagAddr(har.tagged_addr) + har.requested_size);
      Printf("%s", d.Allocation());
      Printf("freed by thread T%u here:\n", t->unique_id());
      Printf("%s", d.Default());
      GetStackTraceFromId(har.free_context_id).Print();

      Printf("%s", d.Allocation());
      Printf("previously allocated by thread T%u here:\n", har.alloc_thread_id);
      Printf("%s", d.Default());
      GetStackTraceFromId(har.alloc_context_id).Print();

      // Print a developer note: the index of this heap object
      // in the thread's deallocation ring buffer.
      Printf("hwasan_dev_note_heap_rb_distance: %zd %zd\n", ring_index + 1,
             flags()->heap_history_size);
      Printf("hwasan_dev_note_num_matching_addrs: %zd\n", num_matching_addrs);
      Printf("hwasan_dev_note_num_matching_addrs_4b: %zd\n",
             num_matching_addrs_4b);

      t->Announce();
      num_descriptions_printed++;
    }
  });

  if (candidate && num_descriptions_printed == 0) {
    ShowHeapOrGlobalCandidate(untagged_addr, candidate, left, right);
    num_descriptions_printed++;
  }

  // Print the remaining threads, as an extra information, 1 line per thread.
  if (flags()->print_live_threads_info)
    hwasanThreadList().VisitAllLiveThreads([&](Thread *t) { t->Announce(); });

  if (!num_descriptions_printed)
    // We exhausted our possibilities. Bail out.
    Printf("HWAddressSanitizer can not describe address in more detail.\n");
  if (num_descriptions_printed > 1) {
    Printf(
        "There are %d potential causes, printed above in order "
        "of likeliness.\n",
        num_descriptions_printed);
  }
}

class InvalidFreeReport : public BaseReport {
 public:
  InvalidFreeReport(StackTrace *stack, uptr tagged_addr)
      : BaseReport(stack, flags()->halt_on_error, tagged_addr, 0) {}
  ~InvalidFreeReport();

 private:
};

InvalidFreeReport::~InvalidFreeReport() {
  tag_t *tag_ptr = nullptr;
  tag_t mem_tag = 0;
  if (MemIsApp(untagged_addr)) {
    tag_ptr = reinterpret_cast<tag_t *>(MemToShadow(untagged_addr));
    if (MemIsShadow(reinterpret_cast<uptr>(tag_ptr)))
      mem_tag = *tag_ptr;
    else
      tag_ptr = nullptr;
  }
  Decorator d;
  Printf("%s", d.Error());
  uptr pc = GetTopPc(stack);
  const char *bug_type = "invalid-free";
  const Thread *thread = GetCurrentThread();
  if (thread) {
    Report("ERROR: %s: %s on address %p at pc %p on thread T%zd\n",
           SanitizerToolName, bug_type, untagged_addr, pc, thread->unique_id());
  } else {
    Report("ERROR: %s: %s on address %p at pc %p on unknown thread\n",
           SanitizerToolName, bug_type, untagged_addr, pc);
  }
  Printf("%s", d.Access());
  if (tag_ptr)
    Printf("tags: %02x/%02x (ptr/mem)\n", ptr_tag, mem_tag);
  Printf("%s", d.Default());

  stack->Print();

  PrintAddressDescription();

  if (tag_ptr)
    PrintTagsAroundAddr(tag_ptr);

  MaybePrintAndroidHelpUrl();
  ReportErrorSummary(bug_type, stack);
}

class TailOverwrittenReport : public BaseReport {
 public:
  explicit TailOverwrittenReport(StackTrace *stack, uptr tagged_addr,
                                 uptr orig_size, const u8 *expected)
      : BaseReport(stack, flags()->halt_on_error, tagged_addr, 0),
        orig_size(orig_size),
        expected(expected) {}
  ~TailOverwrittenReport();

 private:
  uptr orig_size;
  const u8 *expected;
};

TailOverwrittenReport::~TailOverwrittenReport() {
  uptr tail_size = kShadowAlignment - (orig_size % kShadowAlignment);
  u8 actual_expected[kShadowAlignment];
  internal_memcpy(actual_expected, expected, tail_size);
  // Short granule is stashed in the last byte of the magic string. To avoid
  // confusion, make the expected magic string contain the short granule tag.
  if (orig_size % kShadowAlignment != 0) {
    actual_expected[tail_size - 1] = ptr_tag;
  }

  Decorator d;
  Printf("%s", d.Error());
  const char *bug_type = "allocation-tail-overwritten";
  Report("ERROR: %s: %s; heap object [%p,%p) of size %zd\n", SanitizerToolName,
         bug_type, untagged_addr, untagged_addr + orig_size, orig_size);
  Printf("\n%s", d.Default());
  Printf(
      "Stack of invalid access unknown. Issue detected at deallocation "
      "time.\n");
  Printf("%s", d.Allocation());
  Printf("deallocated here:\n");
  Printf("%s", d.Default());
  stack->Print();
  HwasanChunkView chunk = FindHeapChunkByAddress(untagged_addr);
  if (chunk.Beg()) {
    Printf("%s", d.Allocation());
    Printf("allocated here:\n");
    Printf("%s", d.Default());
    GetStackTraceFromId(chunk.GetAllocStackId()).Print();
  }

  InternalScopedString s;
  CHECK_GT(tail_size, 0U);
  CHECK_LT(tail_size, kShadowAlignment);
  u8 *tail = reinterpret_cast<u8*>(untagged_addr + orig_size);
  s.AppendF("Tail contains: ");
  for (uptr i = 0; i < kShadowAlignment - tail_size; i++) s.AppendF(".. ");
  for (uptr i = 0; i < tail_size; i++) s.AppendF("%02x ", tail[i]);
  s.AppendF("\n");
  s.AppendF("Expected:      ");
  for (uptr i = 0; i < kShadowAlignment - tail_size; i++) s.AppendF(".. ");
  for (uptr i = 0; i < tail_size; i++) s.AppendF("%02x ", actual_expected[i]);
  s.AppendF("\n");
  s.AppendF("               ");
  for (uptr i = 0; i < kShadowAlignment - tail_size; i++) s.AppendF("   ");
  for (uptr i = 0; i < tail_size; i++)
    s.AppendF("%s ", actual_expected[i] != tail[i] ? "^^" : "  ");

  s.AppendF(
      "\nThis error occurs when a buffer overflow overwrites memory\n"
      "after a heap object, but within the %zd-byte granule, e.g.\n"
      "   char *x = new char[20];\n"
      "   x[25] = 42;\n"
      "%s does not detect such bugs in uninstrumented code at the time of "
      "write,"
      "\nbut can detect them at the time of free/delete.\n"
      "To disable this feature set HWASAN_OPTIONS=free_checks_tail_magic=0\n",
      kShadowAlignment, SanitizerToolName);
  Printf("%s", s.data());
  GetCurrentThread()->Announce();

  tag_t *tag_ptr = reinterpret_cast<tag_t*>(MemToShadow(untagged_addr));
  PrintTagsAroundAddr(tag_ptr);

  MaybePrintAndroidHelpUrl();
  ReportErrorSummary(bug_type, stack);
}

class TagMismatchReport : public BaseReport {
 public:
  explicit TagMismatchReport(StackTrace *stack, uptr tagged_addr,
                             uptr access_size, bool is_store, bool fatal,
                             uptr *registers_frame)
      : BaseReport(stack, fatal, tagged_addr, access_size),
        access_size(access_size),
        is_store(is_store),
        registers_frame(registers_frame) {}
  ~TagMismatchReport();

 private:
  uptr access_size;
  bool is_store;
  uptr *registers_frame;
};

TagMismatchReport::~TagMismatchReport() {
  Decorator d;
  // TODO: when possible, try to print heap-use-after-free, etc.
  const char *bug_type = "tag-mismatch";
  uptr pc = GetTopPc(stack);
  Printf("%s", d.Error());
  Report("ERROR: %s: %s on address %p at pc %p\n", SanitizerToolName, bug_type,
         untagged_addr, pc);

  Thread *t = GetCurrentThread();

  sptr offset =
      __hwasan_test_shadow(reinterpret_cast<void *>(tagged_addr), access_size);
  CHECK_GE(offset, 0);
  CHECK_LT(offset, static_cast<sptr>(access_size));
  tag_t *tag_ptr =
      reinterpret_cast<tag_t *>(MemToShadow(untagged_addr + offset));
  tag_t mem_tag = *tag_ptr;

  Printf("%s", d.Access());
  if (mem_tag && mem_tag < kShadowAlignment) {
    tag_t *granule_ptr = reinterpret_cast<tag_t *>((untagged_addr + offset) &
                                                   ~(kShadowAlignment - 1));
    // If offset is 0, (untagged_addr + offset) is not aligned to granules.
    // This is the offset of the leftmost accessed byte within the bad granule.
    u8 in_granule_offset = (untagged_addr + offset) & (kShadowAlignment - 1);
    tag_t short_tag = granule_ptr[kShadowAlignment - 1];
    // The first mismatch was a short granule that matched the ptr_tag.
    if (short_tag == ptr_tag) {
      // If the access starts after the end of the short granule, then the first
      // bad byte is the first byte of the access; otherwise it is the first
      // byte past the end of the short granule
      if (mem_tag > in_granule_offset) {
        offset += mem_tag - in_granule_offset;
      }
    }
    Printf(
        "%s of size %zu at %p tags: %02x/%02x(%02x) (ptr/mem) in thread T%zd\n",
        is_store ? "WRITE" : "READ", access_size, untagged_addr, ptr_tag,
        mem_tag, short_tag, t->unique_id());
  } else {
    Printf("%s of size %zu at %p tags: %02x/%02x (ptr/mem) in thread T%zd\n",
           is_store ? "WRITE" : "READ", access_size, untagged_addr, ptr_tag,
           mem_tag, t->unique_id());
  }
  if (offset != 0)
    Printf("Invalid access starting at offset %zu\n", offset);
  Printf("%s", d.Default());

  stack->Print();

  PrintAddressDescription();
  t->Announce();

  PrintTagsAroundAddr(tag_ptr);

  if (registers_frame)
    ReportRegisters(registers_frame, pc);

  MaybePrintAndroidHelpUrl();
  ReportErrorSummary(bug_type, stack);
}
}  // namespace

void ReportInvalidFree(StackTrace *stack, uptr tagged_addr) {
  InvalidFreeReport R(stack, tagged_addr);
}

void ReportTailOverwritten(StackTrace *stack, uptr tagged_addr, uptr orig_size,
                           const u8 *expected) {
  TailOverwrittenReport R(stack, tagged_addr, orig_size, expected);
}

void ReportTagMismatch(StackTrace *stack, uptr tagged_addr, uptr access_size,
                       bool is_store, bool fatal, uptr *registers_frame) {
  TagMismatchReport R(stack, tagged_addr, access_size, is_store, fatal,
                      registers_frame);
}

// See the frame breakdown defined in __hwasan_tag_mismatch (from
// hwasan_tag_mismatch_{aarch64,riscv64}.S).
void ReportRegisters(uptr *frame, uptr pc) {
  Printf("Registers where the failure occurred (pc %p):\n", pc);

  // We explicitly print a single line (4 registers/line) each iteration to
  // reduce the amount of logcat error messages printed. Each Printf() will
  // result in a new logcat line, irrespective of whether a newline is present,
  // and so we wish to reduce the number of Printf() calls we have to make.
#if defined(__aarch64__)
  Printf("    x0  %016llx  x1  %016llx  x2  %016llx  x3  %016llx\n",
       frame[0], frame[1], frame[2], frame[3]);
#elif SANITIZER_RISCV64
  Printf("    sp  %016llx  x1  %016llx  x2  %016llx  x3  %016llx\n",
         reinterpret_cast<u8 *>(frame) + 256, frame[1], frame[2], frame[3]);
#endif
  Printf("    x4  %016llx  x5  %016llx  x6  %016llx  x7  %016llx\n",
       frame[4], frame[5], frame[6], frame[7]);
  Printf("    x8  %016llx  x9  %016llx  x10 %016llx  x11 %016llx\n",
       frame[8], frame[9], frame[10], frame[11]);
  Printf("    x12 %016llx  x13 %016llx  x14 %016llx  x15 %016llx\n",
       frame[12], frame[13], frame[14], frame[15]);
  Printf("    x16 %016llx  x17 %016llx  x18 %016llx  x19 %016llx\n",
       frame[16], frame[17], frame[18], frame[19]);
  Printf("    x20 %016llx  x21 %016llx  x22 %016llx  x23 %016llx\n",
       frame[20], frame[21], frame[22], frame[23]);
  Printf("    x24 %016llx  x25 %016llx  x26 %016llx  x27 %016llx\n",
       frame[24], frame[25], frame[26], frame[27]);
  // hwasan_check* reduces the stack pointer by 256, then __hwasan_tag_mismatch
  // passes it to this function.
#if defined(__aarch64__)
  Printf("    x28 %016llx  x29 %016llx  x30 %016llx   sp %016llx\n", frame[28],
         frame[29], frame[30], reinterpret_cast<u8 *>(frame) + 256);
#elif SANITIZER_RISCV64
  Printf("    x28 %016llx  x29 %016llx  x30 %016llx  x31 %016llx\n", frame[28],
         frame[29], frame[30], frame[31]);
#else
#endif
}

}  // namespace __hwasan

void __hwasan_set_error_report_callback(void (*callback)(const char *)) {
  __hwasan::ScopedReport::SetErrorReportCallback(callback);
}
