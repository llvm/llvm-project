//===--- HLSLResource.cpp - Helper routines for HLSL resources -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides shared routines to help analyze HLSL resources and
// their bindings during Sema and CodeGen.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/HLSLResource.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Type.h"

using namespace clang;

namespace clang {
namespace hlsl {

void EmbeddedResourceNameBuilder::pushBaseName(llvm::StringRef N) {
  pushName(N, FieldDelim);
  Name.append(BaseClassDelim);
}

void EmbeddedResourceNameBuilder::pushName(llvm::StringRef N,
                                           llvm::StringRef Delim) {
  Offsets.push_back(Name.size());
  if (!Name.empty() && !Name.ends_with(BaseClassDelim))
    Name.append(Delim);
  Name.append(N);
}

void EmbeddedResourceNameBuilder::pushArrayIndex(uint64_t Index) {
  llvm::raw_svector_ostream OS(Name);
  Offsets.push_back(Name.size());
  OS << ArrayIndexDelim;
  OS << Index;
}

void EmbeddedResourceNameBuilder::pushBaseNameHierarchy(
    CXXRecordDecl *DerivedRD, CXXRecordDecl *BaseRD) {
  assert(BaseRD != DerivedRD && DerivedRD->isDerivedFrom(BaseRD));
  Offsets.push_back(Name.size());
  Name.append(FieldDelim);
  while (BaseRD != DerivedRD) {
    assert(DerivedRD->getNumBases() == 1 &&
           "HLSL does not support multiple inheritance");
    DerivedRD = DerivedRD->bases_begin()->getType()->getAsCXXRecordDecl();
    assert(DerivedRD && "base class not found");
    Name.append(DerivedRD->getName());
    Name.append(BaseClassDelim);
  }
}

} // namespace hlsl
} // namespace clang
