//===- CodegenNameGenerator.cpp - Codegen name generation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Determines the name that the symbol will get for code generation.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/CodegenNameGenerator.h"
#include "clang/AST/ASTContext.h"

using namespace clang;
using namespace clang::index;

CodegenNameGenerator::CodegenNameGenerator(ASTContext &Ctx)
  : Impl(new ASTNameGenerator(Ctx)) {
}

CodegenNameGenerator::~CodegenNameGenerator() {
}

bool CodegenNameGenerator::writeName(const Decl *D, raw_ostream &OS) {
  return Impl->writeName(D, OS);
}

std::string CodegenNameGenerator::getName(const Decl *D) {
  return Impl->getName(D);
}

std::vector<std::string> CodegenNameGenerator::getAllManglings(const Decl *D) {
  return Impl->getAllManglings(D);
}
