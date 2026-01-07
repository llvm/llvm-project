//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Precompiled header for clangCodeGen. Uses private headers.
///
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "Address.h"
#include "CGBuilder.h"
#include "CGCXXABI.h"
#include "CGValue.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/pch.h"
#include "llvm/IR/pch.h"
#include "llvm/Support/pch.h"
