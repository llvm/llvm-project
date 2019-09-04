//===-- ExpressionSourceCode.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ExpressionSourceCode_h
#define liblldb_ExpressionSourceCode_h

#include "lldb/Expression/Expression.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace lldb_private {

class ExpressionSourceCode {
protected:
  enum Wrapping : bool {
    Wrap = true,
    NoWrap = false,
  };

public:
  bool NeedsWrapping() const { return m_wrap == Wrap; }

  const char *GetName() const { return m_name.c_str(); }

  uint32_t GetNumBodyLines();

  bool GetText(std::string &text, lldb::LanguageType wrapping_language,
               uint32_t language_flags,
               const EvaluateExpressionOptions &options,
               ExecutionContext &exe_ctx, uint32_t &first_body_line) const;

  // Given a string returned by GetText, find the beginning and end of the body
  // passed to CreateWrapped. Return true if the bounds could be found.  This
  // will also work on text with FixItHints applied.
  static bool GetOriginalBodyBounds(std::string transformed_text,
                                    lldb::LanguageType wrapping_language,
                                    size_t &start_loc, size_t &end_loc);

protected:
    ExpressionSourceCode(llvm::StringRef name, llvm::StringRef prefix,
                         llvm::StringRef body, Wrapping wrap)
      : m_name(name.str()), m_prefix(prefix.str()), m_body(body.str()),
        m_num_body_lines(0), m_wrap(wrap) {}

  std::string m_name;
  std::string m_prefix;
  std::string m_body;
  uint32_t m_num_body_lines;
  Wrapping m_wrap;
};

} // namespace lldb_private

#endif
