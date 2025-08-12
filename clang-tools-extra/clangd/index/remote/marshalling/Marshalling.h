//===--- Marshalling.h -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REMOTE_MARSHALLING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REMOTE_MARSHALLING_H

#include "Index.pb.h"
#include "index/Index.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/StringSaver.h"

namespace clang {
namespace clangd {
namespace remote {

// Marshalling provides an interface for translattion between native clangd
// types into the Protobuf-generated classes. Most translations are 1-to-1 and
// wrap variables into appropriate Protobuf types.
//
/// A notable exception is URI translation. Because paths to files are different
/// on indexing machine and client machine
/// ("/remote/machine/projects/llvm-project/llvm/include/HelloWorld.h" versus
/// "/usr/local/username/llvm-project/llvm/include/HelloWorld.h"), they need to
/// be converted appropriately. Remote machine strips the prefix
/// (RemoteIndexRoot) from the absolute path and passes paths relative to the
/// project root over the wire ("include/HelloWorld.h" in this example). The
/// indexed project root is passed to the remote server. Client receives this
/// relative path and constructs a URI that points to the relevant file in the
/// filesystem. The relative path is appended to LocalIndexRoot to construct the
/// full path and build the final URI.
class Marshaller {
public:
  Marshaller() = delete;
  Marshaller(llvm::StringRef RemoteIndexRoot, llvm::StringRef LocalIndexRoot);

  llvm::Expected<clangd::Symbol> fromProtobuf(const Symbol &Message);
  llvm::Expected<clangd::Ref> fromProtobuf(const Ref &Message);
  llvm::Expected<clangd::ContainedRefsResult>
  fromProtobuf(const ContainedRef &Message);
  llvm::Expected<std::pair<clangd::SymbolID, clangd::Symbol>>
  fromProtobuf(const Relation &Message);

  llvm::Expected<clangd::LookupRequest>
  fromProtobuf(const LookupRequest *Message);
  llvm::Expected<clangd::FuzzyFindRequest>
  fromProtobuf(const FuzzyFindRequest *Message);
  llvm::Expected<clangd::RefsRequest> fromProtobuf(const RefsRequest *Message);
  llvm::Expected<clangd::ContainedRefsRequest>
  fromProtobuf(const ContainedRefsRequest *Message);
  llvm::Expected<clangd::RelationsRequest>
  fromProtobuf(const RelationsRequest *Message);

  /// toProtobuf() functions serialize native clangd types and strip IndexRoot
  /// from the file paths specific to indexing machine. fromProtobuf() functions
  /// deserialize clangd types and translate relative paths into machine-native
  /// URIs.
  LookupRequest toProtobuf(const clangd::LookupRequest &From);
  FuzzyFindRequest toProtobuf(const clangd::FuzzyFindRequest &From);
  RefsRequest toProtobuf(const clangd::RefsRequest &From);
  ContainedRefsRequest toProtobuf(const clangd::ContainedRefsRequest &From);
  RelationsRequest toProtobuf(const clangd::RelationsRequest &From);

  llvm::Expected<Symbol> toProtobuf(const clangd::Symbol &From);
  llvm::Expected<Ref> toProtobuf(const clangd::Ref &From);
  llvm::Expected<ContainedRef>
  toProtobuf(const clangd::ContainedRefsResult &From);
  llvm::Expected<Relation> toProtobuf(const clangd::SymbolID &Subject,
                                      const clangd::Symbol &Object);

  /// Translates \p RelativePath into the absolute path and builds URI for the
  /// user machine. This translation happens on the client side with the
  /// \p RelativePath received from remote index server and \p IndexRoot is
  /// provided by the client.
  ///
  /// The relative path passed over the wire has unix slashes.
  llvm::Expected<std::string> relativePathToURI(llvm::StringRef RelativePath);
  /// Translates a URI from the server's backing index to a relative path
  /// suitable to send over the wire to the client.
  llvm::Expected<std::string> uriToRelativePath(llvm::StringRef URI);

private:
  clangd::SymbolLocation::Position fromProtobuf(const Position &Message);
  Position toProtobuf(const clangd::SymbolLocation::Position &Position);
  clang::index::SymbolInfo fromProtobuf(const SymbolInfo &Message);
  SymbolInfo toProtobuf(const clang::index::SymbolInfo &Info);
  llvm::Expected<clangd::SymbolLocation>
  fromProtobuf(const SymbolLocation &Message);
  llvm::Expected<SymbolLocation>
  toProtobuf(const clangd::SymbolLocation &Location);
  llvm::Expected<HeaderWithReferences>
  toProtobuf(const clangd::Symbol::IncludeHeaderWithReferences &IncludeHeader);
  llvm::Expected<clangd::Symbol::IncludeHeaderWithReferences>
  fromProtobuf(const HeaderWithReferences &Message);

  /// RemoteIndexRoot and LocalIndexRoot are absolute paths to the project (on
  /// remote and local machine respectively) and include a trailing slash. One
  /// of them can be missing (if the machines are different they don't know each
  /// other's specifics and will only do one-way translation), but both can not
  /// be missing at the same time.
  std::string RemoteIndexRoot;
  std::string LocalIndexRoot;
  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings;
};

} // namespace remote
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_REMOTE_MARSHALLING_H
