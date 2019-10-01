//===--- URI.h - File URIs with schemes --------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PATHURI_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PATHURI_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Registry.h"

namespace clang {
namespace clangd {

/// A URI describes the location of a source file.
/// In the simplest case, this is a "file" URI that directly encodes the
/// absolute path to a file. More abstract cases are possible: a shared index
/// service might expose repo:// URIs that are relative to the source control
/// root.
///
/// Clangd handles URIs of the form <scheme>:[//<authority>]<body>. It doesn't
/// further split the authority or body into constituent parts (e.g. query
/// strings is included in the body).
class URI {
public:
  URI(llvm::StringRef Scheme, llvm::StringRef Authority, llvm::StringRef Body);

  /// Returns decoded scheme e.g. "https"
  llvm::StringRef scheme() const { return Scheme; }
  /// Returns decoded authority e.g. "reviews.lvm.org"
  llvm::StringRef authority() const { return Authority; }
  /// Returns decoded body e.g. "/D41946"
  llvm::StringRef body() const { return Body; }

  /// Returns a string URI with all components percent-encoded.
  std::string toString() const;

  /// Creates a URI for a file in the given scheme. \p Scheme must be
  /// registered. The URI is percent-encoded.
  static llvm::Expected<URI> create(llvm::StringRef AbsolutePath,
                                    llvm::StringRef Scheme);

  // Similar to above except this picks a registered scheme that works. If none
  // works, this falls back to "file" scheme.
  static URI create(llvm::StringRef AbsolutePath);

  /// This creates a file:// URI for \p AbsolutePath. The path must be absolute.
  static URI createFile(llvm::StringRef AbsolutePath);

  /// Parse a URI string "<scheme>:[//<authority>/]<path>". Percent-encoded
  /// characters in the URI will be decoded.
  static llvm::Expected<URI> parse(llvm::StringRef Uri);

  /// Resolves the absolute path of \p U. If there is no matching scheme, or the
  /// URI is invalid in the scheme, this returns an error.
  ///
  /// \p HintPath A related path, such as the current file or working directory,
  /// which can help disambiguate when the same file exists in many workspaces.
  static llvm::Expected<std::string> resolve(const URI &U,
                                             llvm::StringRef HintPath = "");

  /// Resolves \p AbsPath into a canonical path of its URI, by converting
  /// \p AbsPath to URI and resolving the URI to get th canonical path.
  /// This ensures that paths with the same URI are resolved into consistent
  /// file path.
  static llvm::Expected<std::string> resolvePath(llvm::StringRef AbsPath,
                                                 llvm::StringRef HintPath = "");

  /// Gets the preferred spelling of this file for #include, if there is one,
  /// e.g. <system_header.h>, "path/to/x.h".
  ///
  /// This allows URI schemas to provide their customized include paths.
  ///
  /// Returns an empty string if normal include-shortening based on the absolute
  /// path should be used.
  /// Fails if the URI is not valid in the schema.
  static llvm::Expected<std::string> includeSpelling(const URI &U);

  friend bool operator==(const URI &LHS, const URI &RHS) {
    return std::tie(LHS.Scheme, LHS.Authority, LHS.Body) ==
           std::tie(RHS.Scheme, RHS.Authority, RHS.Body);
  }

  friend bool operator<(const URI &LHS, const URI &RHS) {
    return std::tie(LHS.Scheme, LHS.Authority, LHS.Body) <
           std::tie(RHS.Scheme, RHS.Authority, RHS.Body);
  }

private:
  URI() = default;

  std::string Scheme;
  std::string Authority;
  std::string Body;
};

/// URIScheme is an extension point for teaching clangd to recognize a custom
/// URI scheme. This is expected to be implemented and exposed via the
/// URISchemeRegistry.
class URIScheme {
public:
  virtual ~URIScheme() = default;

  /// Returns the absolute path of the file corresponding to the URI
  /// authority+body in the file system. See URI::resolve for semantics of
  /// \p HintPath.
  virtual llvm::Expected<std::string>
  getAbsolutePath(llvm::StringRef Authority, llvm::StringRef Body,
                  llvm::StringRef HintPath) const = 0;

  virtual llvm::Expected<URI>
  uriFromAbsolutePath(llvm::StringRef AbsolutePath) const = 0;

  /// Returns the include path of the file (e.g. <path>, "path"), which can be
  /// #included directly. See URI::includeSpelling for details.
  virtual llvm::Expected<std::string> getIncludeSpelling(const URI &U) const {
    return ""; // no customized include path for this scheme.
  }
};

/// By default, a "file" scheme is supported where URI paths are always absolute
/// in the file system.
typedef llvm::Registry<URIScheme> URISchemeRegistry;

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_PATHURI_H
