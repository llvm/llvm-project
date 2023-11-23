//===-- sanitizer_symbolizer_markup.h -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file is shared between various sanitizers' runtime libraries.
//
//  Header for the offline markup symbolizer.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_SYMBOLIZER_MARKUP_H
#define SANITIZER_SYMBOLIZER_MARKUP_H

#include "sanitizer_common.h"
#include "sanitizer_stacktrace_printer.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

class MarkupStackTracePrinter : public StackTracePrinter {
 public:
  // We don't support the stack_trace_format flag at all.
  void RenderFrame(InternalScopedString *buffer, const char *format,
                   int frame_no, uptr address, const AddressInfo *info,
                   bool vs_style, const char *strip_path_prefix = "") override;

  bool RenderNeedsSymbolization(const char *format) override;

  // We ignore the format argument to __sanitizer_symbolize_global.
  void RenderData(InternalScopedString *buffer, const char *format,
                  const DataInfo *DI,
                  const char *strip_path_prefix = "") override;

 private:
  void RenderContext(InternalScopedString *buffer);

 protected:
  ~MarkupStackTracePrinter() {}
};

}  // namespace __sanitizer

#endif  // SANITIZER_SYMBOLIZER_MARKUP_H
