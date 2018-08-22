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

class Decorator: public __sanitizer::SanitizerCommonDecorator {
 public:
  Decorator() : SanitizerCommonDecorator() { }
  const char *Access() { return Blue(); }
  const char *Allocation() const { return Magenta(); }
  const char *Origin() const { return Magenta(); }
  const char *Name() const { return Green(); }
};

struct HeapAddressDescription {
  uptr addr;
  u32 alloc_stack_id;
  u32 free_stack_id;

  void Print() const {
    Decorator d;
    if (free_stack_id) {
      Printf("%sfreed here:%s\n", d.Allocation(), d.Default());
      GetStackTraceFromId(free_stack_id).Print();
      Printf("%spreviously allocated here:%s\n", d.Allocation(), d.Default());
    } else {
      Printf("%sallocated here:%s\n", d.Allocation(), d.Default());
    }
    GetStackTraceFromId(alloc_stack_id).Print();
  }
};

bool GetHeapAddressInformation(uptr addr, uptr access_size,
                               HeapAddressDescription *description) {
  HwasanChunkView chunk = FindHeapChunkByAddress(addr);
  if (!chunk.IsValid())
    return false;
  description->addr = addr;
  description->alloc_stack_id = chunk.GetAllocStackId();
  description->free_stack_id = chunk.GetFreeStackId();
  return true;
}

void PrintAddressDescription(uptr addr, uptr access_size) {
  HeapAddressDescription heap_description;
  if (GetHeapAddressInformation(addr, access_size, &heap_description)) {
    heap_description.Print();
    return;
  }
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

  Decorator d;
  Printf("%s", d.Warning());
  Printf("%sTag mismatch in %s%s%s at offset %zu inside [%p, %zu)%s\n",
         d.Warning(), d.Name(), what, d.Warning(), offset, start, size,
         d.Default());
  PrintAddressDescription((uptr)start + offset, 1);
  // if (__sanitizer::Verbosity())
  //   DescribeMemoryRange(start, size);
}

void ReportTagMismatch(StackTrace *stack, uptr addr, uptr access_size,
                       bool is_store) {
  ScopedErrorReportLock l;

  Decorator d;
  Printf("%s", d.Error());
  uptr address = GetAddressFromPointer(addr);
  // TODO: when possible, try to print heap-use-after-free, etc.
  const char *bug_type = "tag-mismatch";
  uptr pc = stack->size ? stack->trace[0] : 0;
  Report("ERROR: %s: %s on address %p at pc %p\n", SanitizerToolName, bug_type, address, pc);

  tag_t ptr_tag = GetTagFromPointer(addr);
  tag_t *tag_ptr = reinterpret_cast<tag_t*>(MEM_TO_SHADOW(address));
  tag_t mem_tag = *tag_ptr;
  Printf("%s", d.Access());
  Printf("%s of size %zu at %p tags: %02x/%02x (ptr/mem)\n",
         is_store ? "WRITE" : "READ", access_size, address, ptr_tag, mem_tag);
  Printf("%s", d.Default());

  stack->Print();

  PrintAddressDescription(address, access_size);

  Printf(
      "Memory tags around the buggy address (one tag corresponds to %zd "
      "bytes):\n", kShadowAlignment);

  const uptr row_len = 16;  // better be power of two.
  const uptr num_rows = 11;
  tag_t *center_row_beg = reinterpret_cast<tag_t *>(
      RoundDownTo(reinterpret_cast<uptr>(tag_ptr), row_len));
  tag_t *beg_row = center_row_beg - row_len * (num_rows / 2);
  tag_t *end_row = center_row_beg + row_len * (num_rows / 2);
  for (tag_t *row = beg_row; row < end_row; row += row_len) {
    Printf("%s", row == center_row_beg ? "=>" : "  ");
    for (uptr i = 0; i < row_len; i++) {
      Printf("%s", row + i == tag_ptr ? "[" : " ");
      Printf("%02x", row[i]);
      Printf("%s", row + i == tag_ptr ? "]" : " ");
    }
    Printf("%s\n", row == center_row_beg ? "<=" : "  ");
  }

  ReportErrorSummary(bug_type, stack);
}

}  // namespace __hwasan
