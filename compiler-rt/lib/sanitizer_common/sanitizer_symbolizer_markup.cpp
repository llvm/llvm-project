//===-- sanitizer_symbolizer_markup.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between various sanitizers' runtime libraries.
//
// Implementation of offline markup symbolizer.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_SYMBOLIZER_MARKUP

#  include "sanitizer_common.h"
#  include "sanitizer_stacktrace_printer.h"
#  include "sanitizer_symbolizer.h"
#  include "sanitizer_symbolizer_markup_constants.h"

namespace __sanitizer {

// This generic support for offline symbolizing is based on the
// Fuchsia port.  We don't do any actual symbolization per se.
// Instead, we emit text containing raw addresses and raw linkage
// symbol names, embedded in Fuchsia's symbolization markup format.
// Fuchsia's logging infrastructure emits enough information about
// process memory layout that a post-processing filter can do the
// symbolization and pretty-print the markup.  See the spec at:
// https://fuchsia.googlesource.com/zircon/+/master/docs/symbolizer_markup.md

// This is used by UBSan for type names, and by ASan for global variable names.
// It's expected to return a static buffer that will be reused on each call.
const char *Symbolizer::Demangle(const char *name) {
  static char buffer[kFormatDemangleMax];
  internal_snprintf(buffer, sizeof(buffer), kFormatDemangle, name);
  return buffer;
}

// This is used mostly for suppression matching.  Making it work
// would enable "interceptor_via_lib" suppressions.  It's also used
// once in UBSan to say "in module ..." in a message that also
// includes an address in the module, so post-processing can already
// pretty-print that so as to indicate the module.
bool Symbolizer::GetModuleNameAndOffsetForPC(uptr pc, const char **module_name,
                                             uptr *module_address) {
  return false;
}

// This is mainly used by hwasan for online symbolization. This isn't needed
// since hwasan can always just dump stack frames for offline symbolization.
bool Symbolizer::SymbolizeFrame(uptr addr, FrameInfo *info) { return false; }

// This is used in some places for suppression checking, which we
// don't really support for Fuchsia.  It's also used in UBSan to
// identify a PC location to a function name, so we always fill in
// the function member with a string containing markup around the PC
// value.
// TODO(mcgrathr): Under SANITIZER_GO, it's currently used by TSan
// to render stack frames, but that should be changed to use
// RenderStackFrame.
SymbolizedStack *Symbolizer::SymbolizePC(uptr addr) {
  SymbolizedStack *s = SymbolizedStack::New(addr);
  char buffer[kFormatFunctionMax];
  internal_snprintf(buffer, sizeof(buffer), kFormatFunction, addr);
  s->info.function = internal_strdup(buffer);
  return s;
}

// Always claim we succeeded, so that RenderDataInfo will be called.
bool Symbolizer::SymbolizeData(uptr addr, DataInfo *info) {
  info->Clear();
  info->start = addr;
  return true;
}

class MarkupStackTracePrinter : public StackTracePrinter {
  // We ignore the format argument to __sanitizer_symbolize_global.
  void RenderData(InternalScopedString *buffer, const char *format,
                  const DataInfo *DI, const char *strip_path_prefix) override {
    buffer->AppendF(kFormatData, DI->start);
  }

  bool RenderNeedsSymbolization(const char *format) override { return false; }

  // We don't support the stack_trace_format flag at all.
  void RenderFrame(InternalScopedString *buffer, const char *format,
                   int frame_no, uptr address, const AddressInfo *info,
                   bool vs_style, const char *strip_path_prefix) override {
    CHECK(!RenderNeedsSymbolization(format));
    buffer->AppendF(kFormatFrame, frame_no, address);
  }

 protected:
  ~MarkupStackTracePrinter();
};

StackTracePrinter *StackTracePrinter::NewStackTracePrinter() {
  return new (GetGlobalLowLevelAllocator()) MarkupStackTracePrinter();
}

Symbolizer *Symbolizer::PlatformInit() {
  return new (symbolizer_allocator_) Symbolizer({});
}

void Symbolizer::LateInitialize() { Symbolizer::GetOrInit(); }

}  // namespace __sanitizer

#endif  // SANITIZER_SYMBOLIZER_MARKUP
