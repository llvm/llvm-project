//===--- ParseHLSLRootSignature.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ParseHLSLRootSignature interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_PARSEHLSLROOTSIGNATURE_H
#define LLVM_CLANG_SEMA_PARSEHLSLROOTSIGNATURE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include "llvm/Frontend/HLSL/HLSLRootSignature.h"

namespace llvm {
namespace hlsl {
namespace root_signature {

class Parser {
public:
  Parser(StringRef Signature, SmallVector<RootElement> *Elements)
      : Buffer(Signature), Elements(Elements) {}

  bool Parse();

private:
  bool ReportError();

  // RootElements parse methods
  bool ParseRootElement();
  bool ParseRootFlags();

  // Enum methods
  template <typename EnumType>
  bool ParseEnum(SmallVector<std::pair<StringLiteral, EnumType>> Mapping,
                 EnumType &Enum);
  bool ParseRootFlag(RootFlags &Flag);

  StringRef Buffer;
  SmallVector<RootElement> *Elements;

  StringRef Token;
};

} // namespace root_signature
} // namespace hlsl
} // namespace llvm

#endif // LLVM_CLANG_SEMA_PARSEHLSLROOTSIGNATURE_H
