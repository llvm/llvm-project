//===--- HLSLRootSignature.h - HLSL Sema Source ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the RootSignatureParsing interface.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_SEMA_HLSLEXTERNALSEMASOURCE_H
#define CLANG_SEMA_HLSLEXTERNALSEMASOURCE_H

#include "llvm/ADT/DenseMap.h"

#include "clang/Sema/ExternalSemaSource.h"

namespace clang {

struct RSTknInfo {
  enum RSTok {
    RootFlags,
    RootConstants,
    RootCBV,
    RootSRV,
    RootUAV,
    DescriptorTable,
    StaticSampler,
    Number,
    Character,
    RootFlag,
    EoF
  };

  RSTknInfo() {}

  RSTok Kind = RSTok::EoF;
  StringRef Text;
};

class RootSignaturParser {

public:
  RootSignaturParser(HLSLRootSignatureAttr *Attr, StringRef Signature)
      : Signature(Signature), Attr(Attr) {}

  void ParseRootDefinition();

private:
  StringRef Signature;
  HLSLRootSignatureAttr *Attr;

  RSTknInfo CurTok;
  std::string IdentifierStr;

  RSTknInfo gettok();

  char nextChar() {
    char resp = Signature[0];
    Signature = Signature.drop_front(1);
    return resp;
  }

  char curChar() { return Signature[0]; }

  RSTknInfo getNextToken() { return CurTok = gettok(); }

  void ParseRootFlag();
};

} // namespace clang
#endif // CLANG_SEMA_HLSLEXTERNALSEMASOURCE_H
