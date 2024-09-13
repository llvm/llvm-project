//===- TableGenBackends.h - Declarations for Clang TableGen Backends ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for all of the Clang TableGen
// backends. A "TableGen backend" is just a function. See
// "$LLVM_ROOT/utils/TableGen/TableGenBackends.h" for more info.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UTILS_TABLEGEN_TABLEGENBACKENDS_H
#define LLVM_CLANG_UTILS_TABLEGEN_TABLEGENBACKENDS_H

#include <string>

namespace llvm {
class raw_ostream;
class RecordKeeper;
} // namespace llvm

namespace clang {

void EmitClangDeclContext(const llvm::RecordKeeper &RK, llvm::raw_ostream &OS);
/**
  @param PriorizeIfSubclassOf These classes should be prioritized in the output.
  This is useful to force enum generation/jump tables/lookup tables to be more
  compact in both size and surrounding code in hot functions. An example use is
  in Decl for classes that inherit from DeclContext, for functions like
  castFromDeclContext.
  */
void EmitClangASTNodes(const llvm::RecordKeeper &RK, llvm::raw_ostream &OS,
                       const std::string &N, const std::string &S,
                       std::string_view PriorizeIfSubclassOf = "");
void EmitClangBasicReader(const llvm::RecordKeeper &Records,
                          llvm::raw_ostream &OS);
void EmitClangBasicWriter(const llvm::RecordKeeper &Records,
                          llvm::raw_ostream &OS);
void EmitClangTypeNodes(const llvm::RecordKeeper &Records,
                        llvm::raw_ostream &OS);
void EmitClangTypeReader(const llvm::RecordKeeper &Records,
                         llvm::raw_ostream &OS);
void EmitClangTypeWriter(const llvm::RecordKeeper &Records,
                         llvm::raw_ostream &OS);
void EmitClangAttrParserStringSwitches(const llvm::RecordKeeper &Records,
                                       llvm::raw_ostream &OS);
void EmitClangAttrSubjectMatchRulesParserStringSwitches(
    const llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitClangAttrClass(const llvm::RecordKeeper &Records,
                        llvm::raw_ostream &OS);
void EmitClangAttrImpl(const llvm::RecordKeeper &Records,
                       llvm::raw_ostream &OS);
void EmitClangAttrList(const llvm::RecordKeeper &Records,
                       llvm::raw_ostream &OS);
void EmitClangAttrSubjectMatchRuleList(const llvm::RecordKeeper &Records,
                                       llvm::raw_ostream &OS);
void EmitClangAttrPCHRead(const llvm::RecordKeeper &Records,
                          llvm::raw_ostream &OS);
void EmitClangAttrPCHWrite(const llvm::RecordKeeper &Records,
                           llvm::raw_ostream &OS);
void EmitClangRegularKeywordAttributeInfo(const llvm::RecordKeeper &Records,
                                          llvm::raw_ostream &OS);
void EmitClangAttrHasAttrImpl(const llvm::RecordKeeper &Records,
                              llvm::raw_ostream &OS);
void EmitClangAttrSpellingListIndex(const llvm::RecordKeeper &Records,
                                    llvm::raw_ostream &OS);
void EmitClangAttrASTVisitor(const llvm::RecordKeeper &Records,
                             llvm::raw_ostream &OS);
void EmitClangAttrTemplateInstantiate(const llvm::RecordKeeper &Records,
                                      llvm::raw_ostream &OS);
void EmitClangAttrParsedAttrList(const llvm::RecordKeeper &Records,
                                 llvm::raw_ostream &OS);
void EmitClangAttrParsedAttrImpl(const llvm::RecordKeeper &Records,
                                 llvm::raw_ostream &OS);
void EmitClangAttrParsedAttrKinds(const llvm::RecordKeeper &Records,
                                  llvm::raw_ostream &OS);
void EmitClangAttrTextNodeDump(const llvm::RecordKeeper &Records,
                               llvm::raw_ostream &OS);
void EmitClangAttrNodeTraverse(const llvm::RecordKeeper &Records,
                               llvm::raw_ostream &OS);
void EmitClangAttrDocTable(const llvm::RecordKeeper &Records,
                           llvm::raw_ostream &OS);

void EmitClangBuiltins(const llvm::RecordKeeper &Records,
                       llvm::raw_ostream &OS);

void EmitClangDiagsDefs(const llvm::RecordKeeper &Records,
                        llvm::raw_ostream &OS, const std::string &Component);
void EmitClangDiagGroups(const llvm::RecordKeeper &Records,
                         llvm::raw_ostream &OS);
void EmitClangDiagsIndexName(const llvm::RecordKeeper &Records,
                             llvm::raw_ostream &OS);

void EmitClangSACheckers(const llvm::RecordKeeper &Records,
                         llvm::raw_ostream &OS);

void EmitClangCommentHTMLTags(const llvm::RecordKeeper &Records,
                              llvm::raw_ostream &OS);
void EmitClangCommentHTMLTagsProperties(const llvm::RecordKeeper &Records,
                                        llvm::raw_ostream &OS);
void EmitClangCommentHTMLNamedCharacterReferences(
    const llvm::RecordKeeper &Records, llvm::raw_ostream &OS);

void EmitClangCommentCommandInfo(const llvm::RecordKeeper &Records,
                                 llvm::raw_ostream &OS);
void EmitClangCommentCommandList(const llvm::RecordKeeper &Records,
                                 llvm::raw_ostream &OS);
void EmitClangOpcodes(const llvm::RecordKeeper &Records, llvm::raw_ostream &OS);

void EmitClangSyntaxNodeList(const llvm::RecordKeeper &Records,
                             llvm::raw_ostream &OS);
void EmitClangSyntaxNodeClasses(const llvm::RecordKeeper &Records,
                                llvm::raw_ostream &OS);

void EmitNeon(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitFP16(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitBF16(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitNeonSema(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitVectorTypes(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitNeonTest(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);

void EmitImmCheckTypes(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitSveHeader(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitSveBuiltins(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitSveBuiltinCG(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitSveTypeFlags(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitSveRangeChecks(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitSveStreamingAttrs(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);

void EmitSmeHeader(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitSmeBuiltins(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitSmeBuiltinCG(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitSmeRangeChecks(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitSmeStreamingAttrs(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitSmeBuiltinZAState(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);

void EmitMveHeader(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitMveBuiltinDef(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitMveBuiltinSema(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitMveBuiltinCG(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitMveBuiltinAliases(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);

void EmitRVVHeader(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitRVVBuiltins(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitRVVBuiltinCG(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitRVVBuiltinSema(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);

void EmitCdeHeader(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitCdeBuiltinDef(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitCdeBuiltinSema(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitCdeBuiltinCG(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);
void EmitCdeBuiltinAliases(llvm::RecordKeeper &Records, llvm::raw_ostream &OS);

void EmitClangAttrDocs(const llvm::RecordKeeper &Records,
                       llvm::raw_ostream &OS);
void EmitClangDiagDocs(const llvm::RecordKeeper &Records,
                       llvm::raw_ostream &OS);
void EmitClangOptDocs(const llvm::RecordKeeper &Records, llvm::raw_ostream &OS);

void EmitClangOpenCLBuiltins(const llvm::RecordKeeper &Records,
                             llvm::raw_ostream &OS);
void EmitClangOpenCLBuiltinHeader(const llvm::RecordKeeper &Records,
                                  llvm::raw_ostream &OS);
void EmitClangOpenCLBuiltinTests(const llvm::RecordKeeper &Records,
                                 llvm::raw_ostream &OS);

void EmitClangDataCollectors(const llvm::RecordKeeper &Records,
                             llvm::raw_ostream &OS);

void EmitTestPragmaAttributeSupportedAttributes(
    const llvm::RecordKeeper &Records, llvm::raw_ostream &OS);

} // end namespace clang

#endif
