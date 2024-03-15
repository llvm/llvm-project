//===- InstallAPI/HeaderFile.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Representations of a library's headers for InstallAPI.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INSTALLAPI_HEADERFILE_H
#define LLVM_CLANG_INSTALLAPI_HEADERFILE_H

#include "clang/Basic/LangStandard.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include <optional>
#include <string>

namespace clang::installapi {
enum class HeaderType {
  /// Unset or unknown type.
  Unknown,
  /// Represents declarations accessible to all clients.
  Public,
  /// Represents declarations accessible to a disclosed set of clients.
  Private,
  /// Represents declarations only accessible as implementation details to the
  /// input library.
  Project,
};

inline StringRef getName(const HeaderType T) {
  switch (T) {
  case HeaderType::Public:
    return "Public";
  case HeaderType::Private:
    return "Private";
  case HeaderType::Project:
    return "Project";
  case HeaderType::Unknown:
    return "Unknown";
  }
  llvm_unreachable("unexpected header type");
}

class HeaderFile {
  /// Full input path to header.
  std::string FullPath;
  /// Access level of header.
  HeaderType Type;
  /// Expected way header will be included by clients.
  std::string IncludeName;
  /// Supported language mode for header.
  std::optional<clang::Language> Language;

public:
  HeaderFile() = delete;
  HeaderFile(StringRef FullPath, HeaderType Type,
             StringRef IncludeName = StringRef(),
             std::optional<clang::Language> Language = std::nullopt)
      : FullPath(FullPath), Type(Type), IncludeName(IncludeName),
        Language(Language) {}

  static llvm::Regex getFrameworkIncludeRule();

  HeaderType getType() const { return Type; }
  StringRef getIncludeName() const { return IncludeName; }
  StringRef getPath() const { return FullPath; }

  bool useIncludeName() const {
    return Type != HeaderType::Project && !IncludeName.empty();
  }

  bool operator==(const HeaderFile &Other) const {
    return std::tie(Type, FullPath, IncludeName, Language) ==
           std::tie(Other.Type, Other.FullPath, Other.IncludeName,
                    Other.Language);
  }
};

/// Assemble expected way header will be included by clients.
/// As in what maps inside the brackets of `#include <IncludeName.h>`
/// For example,
/// "/System/Library/Frameworks/Foo.framework/Headers/Foo.h" returns
/// "Foo/Foo.h"
///
/// \param FullPath Path to the header file which includes the library
/// structure.
std::optional<std::string> createIncludeHeaderName(const StringRef FullPath);
using HeaderSeq = std::vector<HeaderFile>;

} // namespace clang::installapi

#endif // LLVM_CLANG_INSTALLAPI_HEADERFILE_H
