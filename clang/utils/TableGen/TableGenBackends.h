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

void EmitClangAttrClass(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrImpl(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrList(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrPCHRead(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrPCHWrite(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrSpellingList(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrLateParsedList(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrTemplateInstantiate(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrParsedAttrList(RecordKeeper &Records, raw_ostream &OS);
void EmitClangAttrParsedAttrKinds(RecordKeeper &Records, raw_ostream &OS);

void EmitClangDiagsDefs(RecordKeeper &Records, raw_ostream &OS,
                        const std::string &Component);
void EmitClangDiagGroups(RecordKeeper &Records, raw_ostream &OS);
void EmitClangDiagsIndexName(RecordKeeper &Records, raw_ostream &OS);

void EmitClangSACheckers(RecordKeeper &Records, raw_ostream &OS);

void EmitClangCommentHTMLTags(RecordKeeper &Records, raw_ostream &OS);
void EmitClangCommentHTMLTagsProperties(RecordKeeper &Records, raw_ostream &OS);

void EmitClangCommentCommandInfo(RecordKeeper &Records, raw_ostream &OS);

void EmitNeon(RecordKeeper &Records, raw_ostream &OS);
void EmitNeonSema(RecordKeeper &Records, raw_ostream &OS);
void EmitNeonTest(RecordKeeper &Records, raw_ostream &OS);

void EmitOptParser(RecordKeeper &Records, raw_ostream &OS, bool GenDefs);

} // end namespace clang
