//===-- ExpressionSourceCode.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ExpressionSourceCode_h
#define liblldb_ExpressionSourceCode_h

#include "lldb/lldb-enumerations.h"

#include <string>

namespace lldb_private {

class ExpressionSourceCode {
public:
  bool NeedsWrapping() const { return m_wrap; }

  const char *GetName() const { return m_name.c_str(); }

protected:
  ExpressionSourceCode(const char *name, const char *prefix, const char *body,
                       bool wrap)
      : m_name(name), m_prefix(prefix), m_body(body), m_wrap(wrap) {}

  std::string m_name;
  std::string m_prefix;
  std::string m_body;
  bool m_wrap;
};

} // namespace lldb_private

#endif
