//===- ExternalASTSource.cpp - Abstract External AST Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides the default implementation of the ExternalASTSource 
//  interface, which enables construction of AST nodes from some external
//  source.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ExternalASTSource.h"
#include "clang/AST/DeclarationName.h"

using namespace clang;

ExternalASTSource::~ExternalASTSource() { }

void ExternalASTSource::PrintStats() { }

Decl *ExternalASTSource::GetExternalDecl(uint32_t ID) {
  return 0;
}

Selector ExternalASTSource::GetExternalSelector(uint32_t ID) {
  return Selector();
}

uint32_t ExternalASTSource::GetNumExternalSelectors() {
   return 0;
}

Stmt *ExternalASTSource::GetExternalDeclStmt(uint64_t Offset) {
  return 0;
}

CXXBaseSpecifier *
ExternalASTSource::GetExternalCXXBaseSpecifiers(uint64_t Offset) {
  return 0;
}

DeclContextLookupResult 
ExternalASTSource::FindExternalVisibleDeclsByName(const DeclContext *DC,
                                                  DeclarationName Name) {
  return DeclContext::lookup_result();
}

void ExternalASTSource::completeVisibleDeclsMap(const DeclContext *DC) {
}

ExternalLoadResult
ExternalASTSource::FindExternalLexicalDecls(const DeclContext *DC,
                                            bool (*isKindWeWant)(Decl::Kind),
                                         SmallVectorImpl<Decl*> &Result) {
  return ELR_AlreadyLoaded;
}

void ExternalASTSource::getMemoryBufferSizes(MemoryBufferSizes &sizes) const { }
