//===---- URI.h - File URIs with schemes -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "URI.h"
#include "support/Logger.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include <algorithm>

LLVM_INSTANTIATE_REGISTRY(clang::clangd::URISchemeRegistry)

namespace clang {
namespace clangd {
namespace {

bool isWindowsPath(llvm::StringRef Path) {
  return Path.size() > 1 && llvm::isAlpha(Path[0]) && Path[1] == ':';
}

bool isNetworkPath(llvm::StringRef Path) {
  return Path.size() > 2 && Path[0] == Path[1] &&
         llvm::sys::path::is_separator(Path[0]);
}

/// This manages file paths in the file system. All paths in the scheme
/// are absolute (with leading '/').
/// Note that this scheme is hardcoded into the library and not registered in
/// registry.
class FileSystemScheme : public URIScheme {
public:
  llvm::Expected<std::string>
  getAbsolutePath(llvm::StringRef Authority, llvm::StringRef Body,
                  llvm::StringRef /*HintPath*/) const override {
    if (!Body.startswith("/"))
      return error("File scheme: expect body to be an absolute path starting "
                   "with '/': {0}",
                   Body);
    llvm::SmallString<128> Path;
    if (!Authority.empty()) {
      // Windows UNC paths e.g. file://server/share => \\server\share
      ("//" + Authority).toVector(Path);
    } else if (isWindowsPath(Body.substr(1))) {
      // Windows paths e.g. file:///X:/path => X:\path
      Body.consume_front("/");
    }
    Path.append(Body);
    llvm::sys::path::native(Path);
    return std::string(Path);
  }

  llvm::Expected<URI>
  uriFromAbsolutePath(llvm::StringRef AbsolutePath) const override {
    std::string Body;
    llvm::StringRef Authority;
    llvm::StringRef Root = llvm::sys::path::root_name(AbsolutePath);
    if (isNetworkPath(Root)) {
      // Windows UNC paths e.g. \\server\share => file://server/share
      Authority = Root.drop_front(2);
      AbsolutePath.consume_front(Root);
    } else if (isWindowsPath(Root)) {
      // Windows paths e.g. X:\path => file:///X:/path
      Body = "/";
    }
    Body += llvm::sys::path::convert_to_slash(AbsolutePath);
    return URI("file", Authority, Body);
  }
};

llvm::Expected<std::unique_ptr<URIScheme>>
findSchemeByName(llvm::StringRef Scheme) {
  if (Scheme == "file")
    return std::make_unique<FileSystemScheme>();

  for (const auto &URIScheme : URISchemeRegistry::entries()) {
    if (URIScheme.getName() != Scheme)
      continue;
    return URIScheme.instantiate();
  }
  return error("Can't find scheme: {0}", Scheme);
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
void percentEncode(llvm::StringRef Content, std::string &Out) {
  for (unsigned char C : Content)
    if (shouldEscape(C)) {
      Out.push_back('%');
      Out.push_back(llvm::hexdigit(C / 16));
      Out.push_back(llvm::hexdigit(C % 16));
    } else {
      Out.push_back(C);
    }
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
  percentEncode(Scheme, Result);
  Result.push_back(':');
  if (Authority.empty() && Body.empty())
    return Result;
  // If authority if empty, we only print body if it starts with "/"; otherwise,
  // the URI is invalid.
  if (!Authority.empty() || llvm::StringRef(Body).startswith("/"))
  {
    Result.append("//");
    percentEncode(Authority, Result);
  }
  percentEncode(Body, Result);
  return Result;
}

llvm::Expected<URI> URI::parse(llvm::StringRef OrigUri) {
  URI U;
  llvm::StringRef Uri = OrigUri;

  auto Pos = Uri.find(':');
  if (Pos == llvm::StringRef::npos)
    return error("Scheme must be provided in URI: {0}", OrigUri);
  auto SchemeStr = Uri.substr(0, Pos);
  U.Scheme = percentDecode(SchemeStr);
  if (!isValidScheme(U.Scheme))
    return error("Invalid scheme: {0} (decoded: {1})", SchemeStr, U.Scheme);
  Uri = Uri.substr(Pos + 1);
  if (Uri.consume_front("//")) {
    Pos = Uri.find('/');
    U.Authority = percentDecode(Uri.substr(0, Pos));
    Uri = Uri.substr(Pos);
  }
  U.Body = percentDecode(Uri);
  return U;
}

llvm::Expected<std::string> URI::resolve(llvm::StringRef FileURI,
                                         llvm::StringRef HintPath) {
  auto Uri = URI::parse(FileURI);
  if (!Uri)
    return Uri.takeError();
  auto Path = URI::resolve(*Uri, HintPath);
  if (!Path)
    return Path.takeError();
  return *Path;
}

llvm::Expected<URI> URI::create(llvm::StringRef AbsolutePath,
                                llvm::StringRef Scheme) {
  if (!llvm::sys::path::is_absolute(AbsolutePath))
    return error("Not a valid absolute path: {0}", AbsolutePath);
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
  return std::string(AbsPath);
}

llvm::Expected<std::string> URI::includeSpelling(const URI &Uri) {
  auto S = findSchemeByName(Uri.Scheme);
  if (!S)
    return S.takeError();
  return S->get()->getIncludeSpelling(Uri);
}

} // namespace clangd
} // namespace clang
