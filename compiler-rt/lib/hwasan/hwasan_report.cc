//===-- hwasan_report.cc ----------------------------------------------------===//
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
  const char *Allocation() { return Magenta(); }
  const char *Origin() { return Magenta(); }
  const char *Name() { return Green(); }
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
  Printf("%s", d.Warning());
  uptr address = GetAddressFromPointer(addr);
  Printf("%s of size %zu at %p\n", is_store ? "WRITE" : "READ", access_size,
         address);

  tag_t ptr_tag = GetTagFromPointer(addr);
  tag_t mem_tag = *(tag_t *)MEM_TO_SHADOW(address);
  Printf("pointer tag 0x%x\nmemory tag  0x%x\n", ptr_tag, mem_tag);
  Printf("%s", d.Default());

  stack->Print();

  PrintAddressDescription(address, access_size);

  ReportErrorSummary("tag-mismatch", stack);
}


}  // namespace __hwasan
