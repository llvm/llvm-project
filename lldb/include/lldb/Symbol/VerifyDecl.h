//===-- VerifyDecl.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_VariableList_h_
#define lldb_VariableList_h_

#include "lldb/Core/ClangForward.h"

namespace lldb_private {
void VerifyDecl(clang::Decl *decl);
}

#endif
