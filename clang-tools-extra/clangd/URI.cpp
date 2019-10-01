//===---- URI.h - File URIs with schemes -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "URI.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <iomanip>
#include <sstream>

LLVM_INSTANTIATE_REGISTRY(clang::clangd::URISchemeRegistry)

namespace clang {
namespace clangd {
namespace {

inline llvm::Error make_string_error(const llvm::Twine &Message) {
  return llvm::make_error<llvm::StringError>(Message,
                                             llvm::inconvertibleErrorCode());
}

/// \brief This manages file paths in the file system. All paths in the scheme
/// are absolute (with leading '/').
/// Note that this scheme is hardcoded into the library and not registered in
/// registry.
class FileSystemScheme : public URIScheme {
public:
  llvm::Expected<std::string>
  getAbsolutePath(llvm::StringRef /*Authority*/, llvm::StringRef Body,
                  llvm::StringRef /*HintPath*/) const override {
    if (!Body.startswith("/"))
      return make_string_error("File scheme: expect body to be an absolute "
                               "path starting with '/': " +
                               Body);
    // For Windows paths e.g. /X:
    if (Body.size() > 2 && Body[0] == '/' && Body[2] == ':')
      Body.consume_front("/");
    llvm::SmallVector<char, 16> Path(Body.begin(), Body.end());
    llvm::sys::path::native(Path);
    return std::string(Path.begin(), Path.end());
  }

  llvm::Expected<URI>
  uriFromAbsolutePath(llvm::StringRef AbsolutePath) const override {
    std::string Body;
    // For Windows paths e.g. X:
    if (AbsolutePath.size() > 1 && AbsolutePath[1] == ':')
      Body = "/";
    Body += llvm::sys::path::convert_to_slash(AbsolutePath);
    return URI("file", /*Authority=*/"", Body);
  }
};

llvm::Expected<std::unique_ptr<URIScheme>>
findSchemeByName(llvm::StringRef Scheme) {
  if (Scheme == "file")
    return llvm::make_unique<FileSystemScheme>();

  for (auto I = URISchemeRegistry::begin(), E = URISchemeRegistry::end();
       I != E; ++I) {
    if (I->getName() != Scheme)
      continue;
    return I->instantiate();
  }
  return make_string_error("Can't find scheme: " + Scheme);
}

bool shouldEscape(unsigned char C) {
  // Unreserved characters.
  if ((C >= 'a' && C <= 'z') || (C >= 'A' && C <= 'Z') ||
      (C >= '0' && C <= '9'))
    return false;
  switch (C) {
  case '-':
  case '_':
  case '.':
  case '~':
  case '/': // '/' is only reserved when parsing.
  // ':' is only reserved for relative URI paths, which clangd doesn't produce.
  case ':':
    return false;
  }
  return true;
}

/// Encodes a string according to percent-encoding.
/// - Unreserved characters are not escaped.
/// - Reserved characters always escaped with exceptions like '/'.
/// - All other characters are escaped.
std::string percentEncode(llvm::StringRef Content) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  for (unsigned char C : Content)
    if (shouldEscape(C))
      OS << '%' << llvm::format_hex_no_prefix(C, 2, /*Upper = */ true);
    else
      OS << C;

  OS.flush();
  return Result;
}

/// Decodes a string according to percent-encoding.
std::string percentDecode(llvm::StringRef Content) {
  std::string Result;
  for (auto I = Content.begin(), E = Content.end(); I != E; ++I) {
    if (*I != '%') {
      Result += *I;
      continue;
    }
    if (*I == '%' && I + 2 < Content.end() && llvm::isHexDigit(*(I + 1)) &&
        llvm::isHexDigit(*(I + 2))) {
      Result.push_back(llvm::hexFromNibbles(*(I + 1), *(I + 2)));
      I += 2;
    } else
      Result.push_back(*I);
  }
  return Result;
}

bool isValidScheme(llvm::StringRef Scheme) {
  if (Scheme.empty())
    return false;
  if (!llvm::isAlpha(Scheme[0]))
    return false;
  return std::all_of(Scheme.begin() + 1, Scheme.end(), [](char C) {
    return llvm::isAlnum(C) || C == '+' || C == '.' || C == '-';
  });
}

} // namespace

URI::URI(llvm::StringRef Scheme, llvm::StringRef Authority,
         llvm::StringRef Body)
    : Scheme(Scheme), Authority(Authority), Body(Body) {
  assert(!Scheme.empty());
  assert((Authority.empty() || Body.startswith("/")) &&
         "URI body must start with '/' when authority is present.");
}

std::string URI::toString() const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << percentEncode(Scheme) << ":";
  if (Authority.empty() && Body.empty())
    return OS.str();
  // If authority if empty, we only print body if it starts with "/"; otherwise,
  // the URI is invalid.
  if (!Authority.empty() || llvm::StringRef(Body).startswith("/"))
    OS << "//" << percentEncode(Authority);
  OS << percentEncode(Body);
  OS.flush();
  return Result;
}

llvm::Expected<URI> URI::parse(llvm::StringRef OrigUri) {
  URI U;
  llvm::StringRef Uri = OrigUri;

  auto Pos = Uri.find(':');
  if (Pos == llvm::StringRef::npos)
    return make_string_error("Scheme must be provided in URI: " + OrigUri);
  auto SchemeStr = Uri.substr(0, Pos);
  U.Scheme = percentDecode(SchemeStr);
  if (!isValidScheme(U.Scheme))
    return make_string_error(llvm::formatv("Invalid scheme: {0} (decoded: {1})",
                                           SchemeStr, U.Scheme));
  Uri = Uri.substr(Pos + 1);
  if (Uri.consume_front("//")) {
    Pos = Uri.find('/');
    U.Authority = percentDecode(Uri.substr(0, Pos));
    Uri = Uri.substr(Pos);
  }
  U.Body = percentDecode(Uri);
  return U;
}

llvm::Expected<URI> URI::create(llvm::StringRef AbsolutePath,
                                llvm::StringRef Scheme) {
  if (!llvm::sys::path::is_absolute(AbsolutePath))
    return make_string_error("Not a valid absolute path: " + AbsolutePath);
  auto S = findSchemeByName(Scheme);
  if (!S)
    return S.takeError();
  return S->get()->uriFromAbsolutePath(AbsolutePath);
}

URI URI::create(llvm::StringRef AbsolutePath) {
  if (!llvm::sys::path::is_absolute(AbsolutePath))
    llvm_unreachable(
        ("Not a valid absolute path: " + AbsolutePath).str().c_str());
  for (auto &Entry : URISchemeRegistry::entries()) {
    auto URI = Entry.instantiate()->uriFromAbsolutePath(AbsolutePath);
    // For some paths, conversion to different URI schemes is impossible. These
    // should be just skipped.
    if (!URI) {
      // Ignore the error.
      llvm::consumeError(URI.takeError());
      continue;
    }
    return std::move(*URI);
  }
  // Fallback to file: scheme which should work for any paths.
  return URI::createFile(AbsolutePath);
}

URI URI::createFile(llvm::StringRef AbsolutePath) {
  auto U = FileSystemScheme().uriFromAbsolutePath(AbsolutePath);
  if (!U)
    llvm_unreachable(llvm::toString(U.takeError()).c_str());
  return std::move(*U);
}

llvm::Expected<std::string> URI::resolve(const URI &Uri,
                                         llvm::StringRef HintPath) {
  auto S = findSchemeByName(Uri.Scheme);
  if (!S)
    return S.takeError();
  return S->get()->getAbsolutePath(Uri.Authority, Uri.Body, HintPath);
}

llvm::Expected<std::string> URI::resolvePath(llvm::StringRef AbsPath,
                                             llvm::StringRef HintPath) {
  if (!llvm::sys::path::is_absolute(AbsPath))
    llvm_unreachable(("Not a valid absolute path: " + AbsPath).str().c_str());
  for (auto &Entry : URISchemeRegistry::entries()) {
    auto S = Entry.instantiate();
    auto U = S->uriFromAbsolutePath(AbsPath);
    // For some paths, conversion to different URI schemes is impossible. These
    // should be just skipped.
    if (!U) {
      // Ignore the error.
      llvm::consumeError(U.takeError());
      continue;
    }
    return S->getAbsolutePath(U->Authority, U->Body, HintPath);
  }
  // Fallback to file: scheme which doesn't do any canonicalization.
  return AbsPath;
}

llvm::Expected<std::string> URI::includeSpelling(const URI &Uri) {
  auto S = findSchemeByName(Uri.Scheme);
  if (!S)
    return S.takeError();
  return S->get()->getIncludeSpelling(Uri);
}

} // namespace clangd
} // namespace clang
