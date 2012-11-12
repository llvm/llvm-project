//===-- AnalyzerOptions.cpp - Analysis Engine Options -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains special accessors for analyzer configuration options
// with string representations.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;

bool
AnalyzerOptions::mayInlineCXXMemberFunction(CXXInlineableMemberKind K) {
  if (IPAMode < Inlining)
    return false;

  if (!CXXMemberInliningMode) {
    static const char *ModeKey = "c++-inlining";
    
    StringRef ModeStr(Config.GetOrCreateValue(ModeKey,
                                              "methods").getValue());

    CXXInlineableMemberKind &MutableMode =
      const_cast<CXXInlineableMemberKind &>(CXXMemberInliningMode);

    MutableMode = llvm::StringSwitch<CXXInlineableMemberKind>(ModeStr)
      .Case("constructors", CIMK_Constructors)
      .Case("destructors", CIMK_Destructors)
      .Case("none", CIMK_None)
      .Case("methods", CIMK_MemberFunctions)
      .Default(CXXInlineableMemberKind());

    if (!MutableMode) {
      // FIXME: We should emit a warning here about an unknown inlining kind,
      // but the AnalyzerOptions doesn't have access to a diagnostic engine.
      MutableMode = CIMK_None;
    }
  }

  return CXXMemberInliningMode >= K;
}

static StringRef toString(bool b) { return b ? "true" : "false"; }

bool AnalyzerOptions::getBooleanOption(StringRef Name, bool DefaultVal) {
  // FIXME: We should emit a warning here if the value is something other than
  // "true", "false", or the empty string (meaning the default value),
  // but the AnalyzerOptions doesn't have access to a diagnostic engine.
  StringRef V(Config.GetOrCreateValue(Name, toString(DefaultVal)).getValue());
  return llvm::StringSwitch<bool>(V)
      .Case("true", true)
      .Case("false", false)
      .Default(DefaultVal);
}

bool AnalyzerOptions::getBooleanOption(llvm::Optional<bool> &V,
                                       StringRef Name,
                                       bool DefaultVal) {
  if (!V.hasValue())
    V = getBooleanOption(Name, DefaultVal);
  return V.getValue();
}

bool AnalyzerOptions::includeTemporaryDtorsInCFG() {
  return getBooleanOption(IncludeTemporaryDtorsInCFG,
                          "cfg-temporary-dtors",
                          /* Default = */ false);
}

bool AnalyzerOptions::mayInlineCXXStandardLibrary() {
  return getBooleanOption(InlineCXXStandardLibrary,
                          "c++-stdlib-inlining",
                          /*Default=*/true);
}

bool AnalyzerOptions::mayInlineTemplateFunctions() {
  return getBooleanOption(InlineTemplateFunctions,
                          "c++-template-inlining",
                          /*Default=*/true);
}

bool AnalyzerOptions::mayInlineObjCMethod() {
  return getBooleanOption(ObjCInliningMode,
                          "objc-inlining",
                          /* Default = */ true);
}

bool AnalyzerOptions::shouldPruneNullReturnPaths() {
  return getBooleanOption(PruneNullReturnPaths,
                          "suppress-null-return-paths",
                          /* Default = */ true);
}

bool AnalyzerOptions::shouldAvoidSuppressingNullArgumentPaths() {
  return getBooleanOption(AvoidSuppressingNullArgumentPaths,
                          "avoid-suppressing-null-argument-paths",
                          /* Default = */ false);
}

int AnalyzerOptions::getOptionAsInteger(StringRef Name, int DefaultVal) {
  llvm::SmallString<10> StrBuf;
  llvm::raw_svector_ostream OS(StrBuf);
  OS << DefaultVal;
  
  StringRef V(Config.GetOrCreateValue(Name, OS.str()).getValue());
  int Res = DefaultVal;
  bool b = V.getAsInteger(10, Res);
  assert(!b && "analyzer-config option should be numeric");
  (void) b;
  return Res;
}

unsigned AnalyzerOptions::getAlwaysInlineSize() {
  if (!AlwaysInlineSize.hasValue())
    AlwaysInlineSize = getOptionAsInteger("ipa-always-inline-size", 3);
  return AlwaysInlineSize.getValue();
}

unsigned AnalyzerOptions::getGraphTrimInterval() {
  if (!GraphTrimInterval.hasValue())
    GraphTrimInterval = getOptionAsInteger("graph-trim-interval", 1000);
  return GraphTrimInterval.getValue();
}

bool AnalyzerOptions::shouldSynthesizeBodies() {
  return getBooleanOption("faux-bodies", true);
}
