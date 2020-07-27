//===--- Marshalling.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Marshalling.h"
#include "Headers.h"
#include "Index.pb.h"
#include "Protocol.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolID.h"
#include "index/SymbolLocation.h"
#include "index/SymbolOrigin.h"
#include "support/Logger.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

namespace clang {
namespace clangd {
namespace remote {

namespace {

template <typename MessageT>
llvm::Expected<llvm::DenseSet<SymbolID>> getIDs(MessageT *Message) {
  llvm::DenseSet<SymbolID> Result;
  for (const auto &ID : Message->ids()) {
    auto SID = SymbolID::fromStr(StringRef(ID));
    if (!SID)
      return SID.takeError();
    Result.insert(*SID);
  }
  return Result;
}

} // namespace

Marshaller::Marshaller(llvm::StringRef RemoteIndexRoot,
                       llvm::StringRef LocalIndexRoot)
    : Strings(Arena) {
  if (!RemoteIndexRoot.empty()) {
    assert(llvm::sys::path::is_absolute(RemoteIndexRoot));
    assert(RemoteIndexRoot ==
           llvm::sys::path::convert_to_slash(RemoteIndexRoot));
    assert(RemoteIndexRoot.endswith(llvm::sys::path::get_separator()));
    this->RemoteIndexRoot = RemoteIndexRoot.str();
  }
  if (!LocalIndexRoot.empty()) {
    assert(llvm::sys::path::is_absolute(LocalIndexRoot));
    assert(LocalIndexRoot == llvm::sys::path::convert_to_slash(LocalIndexRoot));
    assert(LocalIndexRoot.endswith(llvm::sys::path::get_separator()));
    this->LocalIndexRoot = LocalIndexRoot.str();
  }
  assert(!RemoteIndexRoot.empty() || !LocalIndexRoot.empty());
}

llvm::Expected<clangd::LookupRequest>
Marshaller::fromProtobuf(const LookupRequest *Message) {
  clangd::LookupRequest Req;
  auto IDs = getIDs(Message);
  if (!IDs)
    return IDs.takeError();
  Req.IDs = std::move(*IDs);
  return Req;
}

llvm::Expected<clangd::FuzzyFindRequest>
Marshaller::fromProtobuf(const FuzzyFindRequest *Message) {
  assert(RemoteIndexRoot);
  clangd::FuzzyFindRequest Result;
  Result.Query = Message->query();
  for (const auto &Scope : Message->scopes())
    Result.Scopes.push_back(Scope);
  Result.AnyScope = Message->any_scope();
  if (Message->limit())
    Result.Limit = Message->limit();
  Result.RestrictForCodeCompletion = Message->restricted_for_code_completion();
  for (const auto &Path : Message->proximity_paths()) {
    llvm::SmallString<256> LocalPath = llvm::StringRef(*RemoteIndexRoot);
    llvm::sys::path::append(LocalPath, Path);
    Result.ProximityPaths.push_back(std::string(LocalPath));
  }
  for (const auto &Type : Message->preferred_types())
    Result.ProximityPaths.push_back(Type);
  return Result;
}

llvm::Expected<clangd::RefsRequest>
Marshaller::fromProtobuf(const RefsRequest *Message) {
  clangd::RefsRequest Req;
  auto IDs = getIDs(Message);
  if (!IDs)
    return IDs.takeError();
  Req.IDs = std::move(*IDs);
  Req.Filter = static_cast<RefKind>(Message->filter());
  if (Message->limit())
    Req.Limit = Message->limit();
  return Req;
}

llvm::Optional<clangd::Symbol> Marshaller::fromProtobuf(const Symbol &Message) {
  if (!Message.has_info() || !Message.has_canonical_declaration()) {
    elog("Cannot convert Symbol from protobuf (missing info, definition or "
         "declaration): {0}",
         Message.DebugString());
    return llvm::None;
  }
  clangd::Symbol Result;
  auto ID = SymbolID::fromStr(Message.id());
  if (!ID) {
    elog("Cannot parse SymbolID {0} given protobuf: {1}", ID.takeError(),
         Message.DebugString());
    return llvm::None;
  }
  Result.ID = *ID;
  Result.SymInfo = fromProtobuf(Message.info());
  Result.Name = Message.name();
  Result.Scope = Message.scope();
  if (Message.has_definition()) {
    auto Definition = fromProtobuf(Message.definition());
    if (Definition)
      Result.Definition = *Definition;
  }
  auto Declaration = fromProtobuf(Message.canonical_declaration());
  if (!Declaration) {
    elog("Cannot convert Symbol from protobuf (invalid declaration): {0}",
         Message.DebugString());
    return llvm::None;
  }
  Result.CanonicalDeclaration = *Declaration;
  Result.References = Message.references();
  Result.Origin = static_cast<clangd::SymbolOrigin>(Message.origin());
  Result.Signature = Message.signature();
  Result.TemplateSpecializationArgs = Message.template_specialization_args();
  Result.CompletionSnippetSuffix = Message.completion_snippet_suffix();
  Result.Documentation = Message.documentation();
  Result.ReturnType = Message.return_type();
  Result.Type = Message.type();
  for (const auto &Header : Message.headers()) {
    auto SerializedHeader = fromProtobuf(Header);
    if (SerializedHeader)
      Result.IncludeHeaders.push_back(*SerializedHeader);
    else
      elog("Cannot convert HeaderWithIncludes from protobuf: {0}",
           Header.DebugString());
  }
  Result.Flags = static_cast<clangd::Symbol::SymbolFlag>(Message.flags());
  return Result;
}

llvm::Optional<clangd::Ref> Marshaller::fromProtobuf(const Ref &Message) {
  if (!Message.has_location()) {
    elog("Cannot convert Ref from protobuf (missing location): {0}",
         Message.DebugString());
    return llvm::None;
  }
  clangd::Ref Result;
  auto Location = fromProtobuf(Message.location());
  if (!Location) {
    elog("Cannot convert Ref from protobuf (invalid location): {0}",
         Message.DebugString());
    return llvm::None;
  }
  Result.Location = *Location;
  Result.Kind = static_cast<clangd::RefKind>(Message.kind());
  return Result;
}

LookupRequest Marshaller::toProtobuf(const clangd::LookupRequest &From) {
  LookupRequest RPCRequest;
  for (const auto &SymbolID : From.IDs)
    RPCRequest.add_ids(SymbolID.str());
  return RPCRequest;
}

FuzzyFindRequest Marshaller::toProtobuf(const clangd::FuzzyFindRequest &From) {
  assert(LocalIndexRoot);
  FuzzyFindRequest RPCRequest;
  RPCRequest.set_query(From.Query);
  for (const auto &Scope : From.Scopes)
    RPCRequest.add_scopes(Scope);
  RPCRequest.set_any_scope(From.AnyScope);
  if (From.Limit)
    RPCRequest.set_limit(*From.Limit);
  RPCRequest.set_restricted_for_code_completion(From.RestrictForCodeCompletion);
  for (const auto &Path : From.ProximityPaths) {
    llvm::SmallString<256> RelativePath = llvm::StringRef(Path);
    if (llvm::sys::path::replace_path_prefix(RelativePath, *LocalIndexRoot, ""))
      RPCRequest.add_proximity_paths(llvm::sys::path::convert_to_slash(
          RelativePath, llvm::sys::path::Style::posix));
  }
  for (const auto &Type : From.PreferredTypes)
    RPCRequest.add_preferred_types(Type);
  return RPCRequest;
}

RefsRequest Marshaller::toProtobuf(const clangd::RefsRequest &From) {
  RefsRequest RPCRequest;
  for (const auto &ID : From.IDs)
    RPCRequest.add_ids(ID.str());
  RPCRequest.set_filter(static_cast<uint32_t>(From.Filter));
  if (From.Limit)
    RPCRequest.set_limit(*From.Limit);
  return RPCRequest;
}

llvm::Optional<Symbol> Marshaller::toProtobuf(const clangd::Symbol &From) {
  Symbol Result;
  Result.set_id(From.ID.str());
  *Result.mutable_info() = toProtobuf(From.SymInfo);
  Result.set_name(From.Name.str());
  if (*From.Definition.FileURI) {
    auto Definition = toProtobuf(From.Definition);
    if (!Definition) {
      elog("Can not convert Symbol to protobuf (invalid definition) {0}: {1}",
           From, From.Definition);
      return llvm::None;
    }
    *Result.mutable_definition() = *Definition;
  }
  Result.set_scope(From.Scope.str());
  auto Declaration = toProtobuf(From.CanonicalDeclaration);
  if (!Declaration) {
    elog("Can not convert Symbol to protobuf (invalid declaration) {0}: {1}",
         From, From.CanonicalDeclaration);
    return llvm::None;
  }
  *Result.mutable_canonical_declaration() = *Declaration;
  Result.set_references(From.References);
  Result.set_origin(static_cast<uint32_t>(From.Origin));
  Result.set_signature(From.Signature.str());
  Result.set_template_specialization_args(
      From.TemplateSpecializationArgs.str());
  Result.set_completion_snippet_suffix(From.CompletionSnippetSuffix.str());
  Result.set_documentation(From.Documentation.str());
  Result.set_return_type(From.ReturnType.str());
  Result.set_type(From.Type.str());
  for (const auto &Header : From.IncludeHeaders) {
    auto Serialized = toProtobuf(Header);
    if (!Serialized) {
      elog("Can not convert IncludeHeaderWithReferences to protobuf: {0}",
           Header.IncludeHeader);
      continue;
    }
    auto *NextHeader = Result.add_headers();
    *NextHeader = *Serialized;
  }
  Result.set_flags(static_cast<uint32_t>(From.Flags));
  return Result;
}

llvm::Optional<Ref> Marshaller::toProtobuf(const clangd::Ref &From) {
  Ref Result;
  Result.set_kind(static_cast<uint32_t>(From.Kind));
  auto Location = toProtobuf(From.Location);
  if (!Location) {
    elog("Can not convert Reference to protobuf (invalid location) {0}: {1}",
         From, From.Location);
    return llvm::None;
  }
  *Result.mutable_location() = *Location;
  return Result;
}

llvm::Optional<std::string>
Marshaller::relativePathToURI(llvm::StringRef RelativePath) {
  assert(LocalIndexRoot);
  assert(RelativePath == llvm::sys::path::convert_to_slash(
                             RelativePath, llvm::sys::path::Style::posix));
  if (RelativePath.empty()) {
    return llvm::None;
  }
  if (llvm::sys::path::is_absolute(RelativePath)) {
    return llvm::None;
  }
  llvm::SmallString<256> FullPath = llvm::StringRef(*LocalIndexRoot);
  llvm::sys::path::append(FullPath, RelativePath);
  auto Result = URI::createFile(FullPath);
  return Result.toString();
}

llvm::Optional<std::string> Marshaller::uriToRelativePath(llvm::StringRef URI) {
  assert(RemoteIndexRoot);
  auto ParsedURI = URI::parse(URI);
  if (!ParsedURI) {
    elog("Remote index got bad URI from client {0}: {1}", URI,
         ParsedURI.takeError());
    return llvm::None;
  }
  if (ParsedURI->scheme() != "file") {
    return llvm::None;
  }
  llvm::SmallString<256> Result = ParsedURI->body();
  if (!llvm::sys::path::replace_path_prefix(Result, *RemoteIndexRoot, "")) {
    return llvm::None;
  }
  // Make sure the result has UNIX slashes.
  return llvm::sys::path::convert_to_slash(Result,
                                           llvm::sys::path::Style::posix);
}

clangd::SymbolLocation::Position
Marshaller::fromProtobuf(const Position &Message) {
  clangd::SymbolLocation::Position Result;
  Result.setColumn(static_cast<uint32_t>(Message.column()));
  Result.setLine(static_cast<uint32_t>(Message.line()));
  return Result;
}

Position
Marshaller::toProtobuf(const clangd::SymbolLocation::Position &Position) {
  remote::Position Result;
  Result.set_column(Position.column());
  Result.set_line(Position.line());
  return Result;
}

clang::index::SymbolInfo Marshaller::fromProtobuf(const SymbolInfo &Message) {
  clang::index::SymbolInfo Result;
  Result.Kind = static_cast<clang::index::SymbolKind>(Message.kind());
  Result.SubKind = static_cast<clang::index::SymbolSubKind>(Message.subkind());
  Result.Lang = static_cast<clang::index::SymbolLanguage>(Message.language());
  Result.Properties =
      static_cast<clang::index::SymbolPropertySet>(Message.properties());
  return Result;
}

SymbolInfo Marshaller::toProtobuf(const clang::index::SymbolInfo &Info) {
  SymbolInfo Result;
  Result.set_kind(static_cast<uint32_t>(Info.Kind));
  Result.set_subkind(static_cast<uint32_t>(Info.SubKind));
  Result.set_language(static_cast<uint32_t>(Info.Lang));
  Result.set_properties(static_cast<uint32_t>(Info.Properties));
  return Result;
}

llvm::Optional<clangd::SymbolLocation>
Marshaller::fromProtobuf(const SymbolLocation &Message) {
  clangd::SymbolLocation Location;
  auto URIString = relativePathToURI(Message.file_path());
  if (!URIString)
    return llvm::None;
  Location.FileURI = Strings.save(*URIString).begin();
  Location.Start = fromProtobuf(Message.start());
  Location.End = fromProtobuf(Message.end());
  return Location;
}

llvm::Optional<SymbolLocation>
Marshaller::toProtobuf(const clangd::SymbolLocation &Location) {
  remote::SymbolLocation Result;
  auto RelativePath = uriToRelativePath(Location.FileURI);
  if (!RelativePath)
    return llvm::None;
  *Result.mutable_file_path() = *RelativePath;
  *Result.mutable_start() = toProtobuf(Location.Start);
  *Result.mutable_end() = toProtobuf(Location.End);
  return Result;
}

llvm::Optional<HeaderWithReferences> Marshaller::toProtobuf(
    const clangd::Symbol::IncludeHeaderWithReferences &IncludeHeader) {
  HeaderWithReferences Result;
  Result.set_references(IncludeHeader.References);
  const std::string Header = IncludeHeader.IncludeHeader.str();
  if (isLiteralInclude(Header)) {
    Result.set_header(Header);
    return Result;
  }
  auto RelativePath = uriToRelativePath(Header);
  if (!RelativePath)
    return llvm::None;
  Result.set_header(*RelativePath);
  return Result;
}

llvm::Optional<clangd::Symbol::IncludeHeaderWithReferences>
Marshaller::fromProtobuf(const HeaderWithReferences &Message) {
  std::string Header = Message.header();
  if (Header.front() != '<' && Header.front() != '"') {
    auto URIString = relativePathToURI(Header);
    if (!URIString)
      return llvm::None;
    Header = *URIString;
  }
  return clangd::Symbol::IncludeHeaderWithReferences{Strings.save(Header),
                                                     Message.references()};
}

} // namespace remote
} // namespace clangd
} // namespace clang
