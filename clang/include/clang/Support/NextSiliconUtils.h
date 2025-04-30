#ifndef CLANG_SUPPORT_NEXTSILICONUTILS_H
#define CLANG_SUPPORT_NEXTSILICONUTILS_H

#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Token.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

namespace clang {

namespace ns {

/// NextSilicon (function/loop) mark information
struct PragmaNSMarkInfo {
  /// Result of validity check for the mark
  enum class MarkValidFlags {
    None,     // Invalid mark
    Function, // Valid function mark
    Loop      // Valid loop mark
  };

  Token PragmaName;
  llvm::StringRef Mark;
  MarkValidFlags ValidFlags;
  llvm::ArrayRef<Token> Toks;
  unsigned int ArgumentCount;
};

/// NextSilicon pragma location information
struct PragmaNSLocationInfo {
  Token PragmaName;
  llvm::StringRef NSLocation;
  llvm::ArrayRef<Token> Toks;
  unsigned int ArgumentCount;
};

/// NextSilicon pragma vectorize information
struct PragmaNSVectorizeInfo {
  Token PragmaName;
  llvm::StringRef NSVectorize;
};

/// Check if the flag of ns location is valid.
inline bool isNSLocationValid(llvm::StringRef Location) {
  return llvm::StringSwitch<bool>(Location)
      .Case("grid", true)
      .Case("risc", true)
      .Case("host", true)
      .Default(false);
}

/// Check if the flag of ns mark is valid.
inline PragmaNSMarkInfo::MarkValidFlags
getNSMarkValidFlag(llvm::StringRef Mark) {
  return llvm::StringSwitch<PragmaNSMarkInfo::MarkValidFlags>(Mark)
      .Case("noimport", PragmaNSMarkInfo::MarkValidFlags::Function)
      .Case("handoff", PragmaNSMarkInfo::MarkValidFlags::Function)
      .Case("import_single", PragmaNSMarkInfo::MarkValidFlags::Function)
      // The marked function, and all its callees recursively, should be
      // imported
      .Case("import_recursive", PragmaNSMarkInfo::MarkValidFlags::Function)
      .Case("slot", PragmaNSMarkInfo::MarkValidFlags::Loop)
      .Case("cgid", PragmaNSMarkInfo::MarkValidFlags::Loop)
      .Case("duplication_count", PragmaNSMarkInfo::MarkValidFlags::Loop)
      .Default(PragmaNSMarkInfo::MarkValidFlags::None);
}

/// NextSilicon attributes apply only to function definitions, not declarations.
inline bool isValidFunctionDeclForNSAttr(const FunctionDecl *D) {
  if (D && !D->isThisDeclarationADefinition())
    if (D->getAttr<NextSiliconLocationAttr>() ||
        D->getAttr<NextSiliconMarkAttr>() ||
        D->getAttr<PragmaNextSiliconMarkAttr>() ||
        D->getAttr<PragmaNextSiliconLocationAttr>())
      return false;
  return true;
}

} // end namespace ns
} // end namespace clang

#endif // CLANG_SUPPORT_NEXTSILICONUTILS_H
