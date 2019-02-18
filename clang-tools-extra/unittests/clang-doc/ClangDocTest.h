//===-- clang-doc/ClangDocTest.h ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_DOC_CLANGDOCTEST_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_DOC_CLANGDOCTEST_H

#include "ClangDocTest.h"
#include "Representation.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

using EmittedInfoList = std::vector<std::unique_ptr<Info>>;

static const SymbolID EmptySID = SymbolID();
static const SymbolID NonEmptySID =
    SymbolID{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

NamespaceInfo *InfoAsNamespace(Info *I);
RecordInfo *InfoAsRecord(Info *I);
FunctionInfo *InfoAsFunction(Info *I);
EnumInfo *InfoAsEnum(Info *I);

// Unlike the operator==, these functions explicitly does not check USRs, as
// that may change and it would be better to not rely on its implementation.
void CheckReference(Reference &Expected, Reference &Actual);
void CheckTypeInfo(TypeInfo *Expected, TypeInfo *Actual);
void CheckFieldTypeInfo(FieldTypeInfo *Expected, FieldTypeInfo *Actual);
void CheckMemberTypeInfo(MemberTypeInfo *Expected, MemberTypeInfo *Actual);

// This function explicitly does not check USRs, as that may change and it would
// be better to not rely on its implementation.
void CheckBaseInfo(Info *Expected, Info *Actual);
void CheckSymbolInfo(SymbolInfo *Expected, SymbolInfo *Actual);
void CheckFunctionInfo(FunctionInfo *Expected, FunctionInfo *Actual);
void CheckEnumInfo(EnumInfo *Expected, EnumInfo *Actual);
void CheckNamespaceInfo(NamespaceInfo *Expected, NamespaceInfo *Actual);
void CheckRecordInfo(RecordInfo *Expected, RecordInfo *Actual);

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_DOC_CLANGDOCTEST_H
