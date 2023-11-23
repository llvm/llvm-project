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
// This generic support for offline symbolizing is based on the
// Fuchsia port.  We don't do any actual symbolization per se.
// Instead, we emit text containing raw addresses and raw linkage
// symbol names, embedded in Fuchsia's symbolization markup format.
// See the spec at:
// https://llvm.org/docs/SymbolizerMarkupFormat.html
//===----------------------------------------------------------------------===//

#include "sanitizer_symbolizer_markup.h"

#include "sanitizer_common.h"
#include "sanitizer_stacktrace_printer.h"
#include "sanitizer_symbolizer.h"
#include "sanitizer_symbolizer_markup_constants.h"

namespace __sanitizer {

void MarkupStackTracePrinter::RenderData(InternalScopedString *buffer,
                                         const char *format, const DataInfo *DI,
                                         const char *strip_path_prefix) {
  buffer->AppendF(kFormatData, DI->start);
}

bool MarkupStackTracePrinter::RenderNeedsSymbolization(const char *format) {
  return false;
}

// We don't support the stack_trace_format flag at all.
void MarkupStackTracePrinter::RenderFrame(InternalScopedString *buffer,
                                          const char *format, int frame_no,
                                          uptr address, const AddressInfo *info,
                                          bool vs_style,
                                          const char *strip_path_prefix) {
  CHECK(!RenderNeedsSymbolization(format));
  buffer->AppendF(kFormatFrame, frame_no, address);
}

bool MarkupSymbolizerTool::SymbolizePC(uptr addr, SymbolizedStack *stack) {
  char buffer[kFormatFunctionMax];
  internal_snprintf(buffer, sizeof(buffer), kFormatFunction, addr);
  stack->info.function = internal_strdup(buffer);
  return true;
}

bool MarkupSymbolizerTool::SymbolizeData(uptr addr, DataInfo *info) {
  info->Clear();
  info->start = addr;
  return true;
}

const char *MarkupSymbolizerTool::Demangle(const char *name) {
  static char buffer[kFormatDemangleMax];
  internal_snprintf(buffer, sizeof(buffer), kFormatDemangle, name);
  return buffer;
}

}  // namespace __sanitizer
