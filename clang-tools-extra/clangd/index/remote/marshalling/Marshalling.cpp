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
#include "index/Index.h"
#include "index/Ref.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolID.h"
#include "index/SymbolLocation.h"
#include "index/SymbolOrigin.h"
#include "support/Logger.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"

namespace clang {
namespace clangd {
namespace remote {

using llvm::sys::path::append;
using llvm::sys::path::convert_to_slash;
using llvm::sys::path::is_absolute;
using llvm::sys::path::replace_path_prefix;
using llvm::sys::path::Style;

namespace {

template <typename IDRange>
llvm::Expected<llvm::DenseSet<SymbolID>> getIDs(IDRange IDs) {
  llvm::DenseSet<SymbolID> Result;
  for (const auto &ID : IDs) {
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
  llvm::StringRef PosixSeparator = get_separator(Style::posix);
  if (!RemoteIndexRoot.empty()) {
    assert(is_absolute(RemoteIndexRoot));
    this->RemoteIndexRoot = convert_to_slash(RemoteIndexRoot, Style::windows);
    llvm::StringRef Path(this->RemoteIndexRoot);
    if (!is_separator(this->RemoteIndexRoot.back(), Style::posix))
      this->RemoteIndexRoot += PosixSeparator;
  }
  if (!LocalIndexRoot.empty()) {
    assert(is_absolute(LocalIndexRoot));
    this->LocalIndexRoot = convert_to_slash(LocalIndexRoot, Style::windows);
    llvm::StringRef Path(this->LocalIndexRoot);
    if (!is_separator(this->LocalIndexRoot.back(), Style::posix))
      this->LocalIndexRoot += PosixSeparator;
  }
  assert(!RemoteIndexRoot.empty() || !LocalIndexRoot.empty());
}

llvm::Expected<clangd::LookupRequest>
Marshaller::fromProtobuf(const LookupRequest *Message) {
  clangd::LookupRequest Req;
  auto IDs = getIDs(Message->ids());
  if (!IDs)
    return IDs.takeError();
  Req.IDs = std::move(*IDs);
  return Req;
}

llvm::Expected<clangd::FuzzyFindRequest>
Marshaller::fromProtobuf(const FuzzyFindRequest *Message) {
  assert(!RemoteIndexRoot.empty());
  clangd::FuzzyFindRequest Result;
  Result.Query = Message->query();
  for (const auto &Scope : Message->scopes())
    Result.Scopes.push_back(Scope);
  Result.AnyScope = Message->any_scope();
  if (Message->limit())
    Result.Limit = Message->limit();
  Result.RestrictForCodeCompletion = Message->restricted_for_code_completion();
  for (const auto &Path : Message->proximity_paths()) {
    llvm::SmallString<256> LocalPath = llvm::StringRef(RemoteIndexRoot);
    append(LocalPath, Path);
    // FuzzyFindRequest requires proximity paths to have platform-native format
    // in order for SymbolIndex to process the query correctly.
    llvm::sys::path::native(LocalPath);
    Result.ProximityPaths.push_back(std::string(LocalPath));
  }
  for (const auto &Type : Message->preferred_types())
    Result.ProximityPaths.push_back(Type);
  return Result;
}

llvm::Expected<clangd::RefsRequest>
Marshaller::fromProtobuf(const RefsRequest *Message) {
  clangd::RefsRequest Req;
  auto IDs = getIDs(Message->ids());
  if (!IDs)
    return IDs.takeError();
  Req.IDs = std::move(*IDs);
  if (Message->has_filter())
    Req.Filter = static_cast<clangd::RefKind>(Message->filter());
  else
    Req.Filter = clangd::RefKind::All;
  if (Message->limit())
    Req.Limit = Message->limit();
  return Req;
}

llvm::Expected<clangd::RelationsRequest>
Marshaller::fromProtobuf(const RelationsRequest *Message) {
  clangd::RelationsRequest Req;
  auto IDs = getIDs(Message->subjects());
  if (!IDs)
    return IDs.takeError();
  Req.Subjects = std::move(*IDs);
  if (!Message->has_predicate())
    return error("RelationsRequest requires RelationKind predicate.");
  Req.Predicate = static_cast<RelationKind>(Message->predicate());
  if (Message->limit())
    Req.Limit = Message->limit();
  return Req;
}

llvm::Expected<clangd::Symbol> Marshaller::fromProtobuf(const Symbol &Message) {
  if (!Message.has_info() || !Message.has_canonical_declaration())
    return error("Missing info or declaration.");
  clangd::Symbol Result;
  auto ID = SymbolID::fromStr(Message.id());
  if (!ID)
    return ID.takeError();
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
  if (!Declaration)
    return Declaration.takeError();
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
    if (!SerializedHeader)
      return SerializedHeader.takeError();
    Result.IncludeHeaders.push_back(*SerializedHeader);
  }
  Result.Flags = static_cast<clangd::Symbol::SymbolFlag>(Message.flags());
  return Result;
}

llvm::Expected<clangd::Ref> Marshaller::fromProtobuf(const Ref &Message) {
  if (!Message.has_location())
    return error("Missing location.");
  clangd::Ref Result;
  auto Location = fromProtobuf(Message.location());
  if (!Location)
    return Location.takeError();
  Result.Location = *Location;
  Result.Kind = static_cast<RefKind>(Message.kind());
  return Result;
}

llvm::Expected<std::pair<clangd::SymbolID, clangd::Symbol>>
Marshaller::fromProtobuf(const Relation &Message) {
  auto SubjectID = SymbolID::fromStr(Message.subject_id());
  if (!SubjectID)
    return SubjectID.takeError();
  if (!Message.has_object())
    return error("Missing Object.");
  auto Object = fromProtobuf(Message.object());
  if (!Object)
    return Object.takeError();
  return std::make_pair(*SubjectID, *Object);
}

LookupRequest Marshaller::toProtobuf(const clangd::LookupRequest &From) {
  LookupRequest RPCRequest;
  for (const auto &SymbolID : From.IDs)
    RPCRequest.add_ids(SymbolID.str());
  return RPCRequest;
}

FuzzyFindRequest Marshaller::toProtobuf(const clangd::FuzzyFindRequest &From) {
  assert(!LocalIndexRoot.empty());
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
    if (replace_path_prefix(RelativePath, LocalIndexRoot, ""))
      RPCRequest.add_proximity_paths(
          convert_to_slash(RelativePath, Style::windows));
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

RelationsRequest Marshaller::toProtobuf(const clangd::RelationsRequest &From) {
  RelationsRequest RPCRequest;
  for (const auto &ID : From.Subjects)
    RPCRequest.add_subjects(ID.str());
  RPCRequest.set_predicate(static_cast<uint32_t>(From.Predicate));
  if (From.Limit)
    RPCRequest.set_limit(*From.Limit);
  return RPCRequest;
}

llvm::Expected<Symbol> Marshaller::toProtobuf(const clangd::Symbol &From) {
  Symbol Result;
  Result.set_id(From.ID.str());
  *Result.mutable_info() = toProtobuf(From.SymInfo);
  Result.set_name(From.Name.str());
  if (*From.Definition.FileURI) {
    auto Definition = toProtobuf(From.Definition);
    if (!Definition)
      return Definition.takeError();
    *Result.mutable_definition() = *Definition;
  }
  Result.set_scope(From.Scope.str());
  auto Declaration = toProtobuf(From.CanonicalDeclaration);
  if (!Declaration)
    return Declaration.takeError();
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
    if (!Serialized)
      return Serialized.takeError();
    auto *NextHeader = Result.add_headers();
    *NextHeader = *Serialized;
  }
  Result.set_flags(static_cast<uint32_t>(From.Flags));
  return Result;
}

llvm::Expected<Ref> Marshaller::toProtobuf(const clangd::Ref &From) {
  Ref Result;
  Result.set_kind(static_cast<uint32_t>(From.Kind));
  auto Location = toProtobuf(From.Location);
  if (!Location)
    return Location.takeError();
  *Result.mutable_location() = *Location;
  return Result;
}

llvm::Expected<Relation> Marshaller::toProtobuf(const clangd::SymbolID &Subject,
                                                const clangd::Symbol &Object) {
  Relation Result;
  *Result.mutable_subject_id() = Subject.str();
  auto SerializedObject = toProtobuf(Object);
  if (!SerializedObject)
    return SerializedObject.takeError();
  *Result.mutable_object() = *SerializedObject;
  return Result;
}

llvm::Expected<std::string>
Marshaller::relativePathToURI(llvm::StringRef RelativePath) {
  assert(!LocalIndexRoot.empty());
  assert(RelativePath == convert_to_slash(RelativePath));
  if (RelativePath.empty())
    return error("Empty relative path.");
  if (is_absolute(RelativePath, Style::posix))
    return error("RelativePath '{0}' is absolute.", RelativePath);
  llvm::SmallString<256> FullPath = llvm::StringRef(LocalIndexRoot);
  append(FullPath, RelativePath);
  auto Result = URI::createFile(FullPath);
  return Result.toString();
}

llvm::Expected<std::string> Marshaller::uriToRelativePath(llvm::StringRef URI) {
  assert(!RemoteIndexRoot.empty());
  auto ParsedURI = URI::parse(URI);
  if (!ParsedURI)
    return ParsedURI.takeError();
  if (ParsedURI->scheme() != "file")
    return error("Can not use URI schemes other than file, given: '{0}'.", URI);
  llvm::SmallString<256> Result = ParsedURI->body();
  llvm::StringRef Path(Result);
  // Check for Windows paths (URI=file:///X:/path => Body=/X:/path)
  if (is_absolute(Path.substr(1), Style::windows))
    Result = Path.drop_front();
  if (!replace_path_prefix(Result, RemoteIndexRoot, ""))
    return error("File path '{0}' doesn't start with '{1}'.", Result.str(),
                 RemoteIndexRoot);
  assert(Result == convert_to_slash(Result, Style::windows));
  return std::string(Result);
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

llvm::Expected<clangd::SymbolLocation>
Marshaller::fromProtobuf(const SymbolLocation &Message) {
  clangd::SymbolLocation Location;
  auto URIString = relativePathToURI(Message.file_path());
  if (!URIString)
    return URIString.takeError();
  Location.FileURI = Strings.save(*URIString).begin();
  Location.Start = fromProtobuf(Message.start());
  Location.End = fromProtobuf(Message.end());
  return Location;
}

llvm::Expected<SymbolLocation>
Marshaller::toProtobuf(const clangd::SymbolLocation &Location) {
  remote::SymbolLocation Result;
  auto RelativePath = uriToRelativePath(Location.FileURI);
  if (!RelativePath)
    return RelativePath.takeError();
  *Result.mutable_file_path() = *RelativePath;
  *Result.mutable_start() = toProtobuf(Location.Start);
  *Result.mutable_end() = toProtobuf(Location.End);
  return Result;
}

llvm::Expected<HeaderWithReferences> Marshaller::toProtobuf(
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
    return RelativePath.takeError();
  Result.set_header(*RelativePath);
  return Result;
}

llvm::Expected<clangd::Symbol::IncludeHeaderWithReferences>
Marshaller::fromProtobuf(const HeaderWithReferences &Message) {
  std::string Header = Message.header();
  if (!isLiteralInclude(Header)) {
    auto URIString = relativePathToURI(Header);
    if (!URIString)
      return URIString.takeError();
    Header = *URIString;
  }
  return clangd::Symbol::IncludeHeaderWithReferences{Strings.save(Header),
                                                     Message.references()};
}

} // namespace remote
} // namespace clangd
} // namespace clang
