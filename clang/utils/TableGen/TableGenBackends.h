//===- TableGenBackends.h - Declarations for Clang TableGen Backends ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
}

using llvm::raw_ostream;
using llvm::RecordKeeper;

namespace clang {

void EmitClangDeclContext(RecordKeeper &RK, raw_ostream &OS);
void EmitClangASTNodes(RecordKeeper &RK, raw_ostream &OS,
                       const std::string &N, const std::string &S);

void EmitClangAttrParserStringSwitches(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrSubjectMatchRulesParserStringSwitches(RecordKeeper &Records,
                                                        raw_ostream &OS);
void EmitClangAttrClass(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrImpl(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrList(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrSubjectMatchRuleList(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrPCHRead(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrPCHWrite(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrHasAttrImpl(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrSpellingListIndex(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrASTVisitor(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrTemplateInstantiate(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrParsedAttrList(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrParsedAttrImpl(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrParsedAttrKinds(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrDump(RecordKeeper &Records, raw_ostream &OS);

void EmitClangDiagsDefs(RecordKeeper &Records, raw_ostream &OS,
                        const std::string &Component);
void EmitClangDiagGroups(RecordKeeper &Records, raw_ostream &OS);
void EmitClangDiagsIndexName(RecordKeeper &Records, raw_ostream &OS);

void EmitClangSACheckers(RecordKeeper &Records, raw_ostream &OS);

void EmitClangCommentHTMLTags(RecordKeeper &Records, raw_ostream &OS);
void EmitClangCommentHTMLTagsProperties(RecordKeeper &Records, raw_ostream &OS);
void EmitClangCommentHTMLNamedCharacterReferences(RecordKeeper &Records, raw_ostream &OS);

void EmitClangCommentCommandInfo(RecordKeeper &Records, raw_ostream &OS);
void EmitClangCommentCommandList(RecordKeeper &Records, raw_ostream &OS);

void EmitNeon(RecordKeeper &Records, raw_ostream &OS);
void EmitFP16(RecordKeeper &Records, raw_ostream &OS);
void EmitNeonSema(RecordKeeper &Records, raw_ostream &OS);
void EmitNeonTest(RecordKeeper &Records, raw_ostream &OS);
void EmitNeon2(RecordKeeper &Records, raw_ostream &OS);
void EmitNeonSema2(RecordKeeper &Records, raw_ostream &OS);
void EmitNeonTest2(RecordKeeper &Records, raw_ostream &OS);

void EmitClangAttrDocs(RecordKeeper &Records, raw_ostream &OS);
void EmitClangDiagDocs(RecordKeeper &Records, raw_ostream &OS);
void EmitClangOptDocs(RecordKeeper &Records, raw_ostream &OS);

void EmitClangDataCollectors(RecordKeeper &Records, raw_ostream &OS);

void EmitTestPragmaAttributeSupportedAttributes(RecordKeeper &Records,
                                                raw_ostream &OS);

} // end namespace clang

#endif
