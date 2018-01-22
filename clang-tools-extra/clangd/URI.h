//===--- URI.h - File URIs with schemes --------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
class FileURI {
public:
  /// Returns decoded scheme e.g. "https"
  llvm::StringRef scheme() const { return Scheme; }
  /// Returns decoded authority e.g. "reviews.lvm.org"
  llvm::StringRef authority() const { return Authority; }
  /// Returns decoded body e.g. "/D41946"
  llvm::StringRef body() const { return Body; }

  /// Returns a string URI with all components percent-encoded.
  std::string toString() const;

  /// Create a FileURI from unescaped scheme+authority+body.
  static llvm::Expected<FileURI> create(llvm::StringRef Scheme,
                                        llvm::StringRef Authority,
                                        llvm::StringRef Body);

  /// Creates a FileURI for a file in the given scheme. \p Scheme must be
  /// registered. The URI is percent-encoded.
  static llvm::Expected<FileURI> create(llvm::StringRef AbsolutePath,
                                        llvm::StringRef Scheme = "file");

  /// Parse a URI string "<scheme>:[//<authority>/]<path>". Percent-encoded
  /// characters in the URI will be decoded.
  static llvm::Expected<FileURI> parse(llvm::StringRef Uri);

  /// Resolves the absolute path of \p U. If there is no matching scheme, or the
  /// URI is invalid in the scheme, this returns an error.
  ///
  /// \p HintPath A related path, such as the current file or working directory,
  /// which can help disambiguate when the same file exists in many workspaces.
  static llvm::Expected<std::string> resolve(const FileURI &U,
                                             llvm::StringRef HintPath = "");

  friend bool operator==(const FileURI &LHS, const FileURI &RHS) {
    return std::tie(LHS.Scheme, LHS.Authority, LHS.Body) ==
           std::tie(RHS.Scheme, RHS.Authority, RHS.Body);
  }

private:
  FileURI() = default;

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
  /// authority+body in the file system. See FileURI::resolve for semantics of
  /// \p HintPath.
  virtual llvm::Expected<std::string>
  getAbsolutePath(llvm::StringRef Authority, llvm::StringRef Body,
                  llvm::StringRef HintPath) const = 0;

  virtual llvm::Expected<FileURI>
  uriFromAbsolutePath(llvm::StringRef AbsolutePath) const = 0;
};

/// By default, a "file" scheme is supported where URI paths are always absolute
/// in the file system.
typedef llvm::Registry<URIScheme> URISchemeRegistry;

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_PATHURI_H
