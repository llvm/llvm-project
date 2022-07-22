///===-- Representation.cpp - ClangDoc Representation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the merging of different types of infos. The data in the
// calling Info is preserved during a merge unless that field is empty or
// default. In that case, the data from the parameter Info is used to replace
// the empty or default data.
//
// For most fields, the first decl seen provides the data. Exceptions to this
// include the location and description fields, which are collections of data on
// all decls related to a given definition. All other fields are ignored in new
// decls unless the first seen decl didn't, for whatever reason, incorporate
// data on that field (e.g. a forward declared class wouldn't have information
// on members on the forward declaration, but would have the class name).
//
//===----------------------------------------------------------------------===//
#include "Representation.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace doc {

namespace {

const SymbolID EmptySID = SymbolID();

template <typename T>
llvm::Expected<std::unique_ptr<Info>>
reduce(std::vector<std::unique_ptr<Info>> &Values) {
  if (Values.empty() || !Values[0])
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no value to reduce");
  std::unique_ptr<Info> Merged = std::make_unique<T>(Values[0]->USR);
  T *Tmp = static_cast<T *>(Merged.get());
  for (auto &I : Values)
    Tmp->merge(std::move(*static_cast<T *>(I.get())));
  return std::move(Merged);
}

// Return the index of the matching child in the vector, or -1 if merge is not
// necessary.
template <typename T>
int getChildIndexIfExists(std::vector<T> &Children, T &ChildToMerge) {
  for (unsigned long I = 0; I < Children.size(); I++) {
    if (ChildToMerge.USR == Children[I].USR)
      return I;
  }
  return -1;
}

void reduceChildren(std::vector<Reference> &Children,
                    std::vector<Reference> &&ChildrenToMerge) {
  for (auto &ChildToMerge : ChildrenToMerge) {
    int mergeIdx = getChildIndexIfExists(Children, ChildToMerge);
    if (mergeIdx == -1) {
      Children.push_back(std::move(ChildToMerge));
      continue;
    }
    Children[mergeIdx].merge(std::move(ChildToMerge));
  }
}

void reduceChildren(std::vector<FunctionInfo> &Children,
                    std::vector<FunctionInfo> &&ChildrenToMerge) {
  for (auto &ChildToMerge : ChildrenToMerge) {
    int mergeIdx = getChildIndexIfExists(Children, ChildToMerge);
    if (mergeIdx == -1) {
      Children.push_back(std::move(ChildToMerge));
      continue;
    }
    Children[mergeIdx].merge(std::move(ChildToMerge));
  }
}

void reduceChildren(std::vector<EnumInfo> &Children,
                    std::vector<EnumInfo> &&ChildrenToMerge) {
  for (auto &ChildToMerge : ChildrenToMerge) {
    int mergeIdx = getChildIndexIfExists(Children, ChildToMerge);
    if (mergeIdx == -1) {
      Children.push_back(std::move(ChildToMerge));
      continue;
    }
    Children[mergeIdx].merge(std::move(ChildToMerge));
  }
}

} // namespace

// Dispatch function.
llvm::Expected<std::unique_ptr<Info>>
mergeInfos(std::vector<std::unique_ptr<Info>> &Values) {
  if (Values.empty() || !Values[0])
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no info values to merge");

  switch (Values[0]->IT) {
  case InfoType::IT_namespace:
    return reduce<NamespaceInfo>(Values);
  case InfoType::IT_record:
    return reduce<RecordInfo>(Values);
  case InfoType::IT_enum:
    return reduce<EnumInfo>(Values);
  case InfoType::IT_function:
    return reduce<FunctionInfo>(Values);
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unexpected info type");
  }
}

static llvm::SmallString<64>
calculateRelativeFilePath(const InfoType &Type, const StringRef &Path,
                          const StringRef &Name, const StringRef &CurrentPath) {
  llvm::SmallString<64> FilePath;

  if (CurrentPath != Path) {
    // iterate back to the top
    for (llvm::sys::path::const_iterator I =
             llvm::sys::path::begin(CurrentPath);
         I != llvm::sys::path::end(CurrentPath); ++I)
      llvm::sys::path::append(FilePath, "..");
    llvm::sys::path::append(FilePath, Path);
  }

  // Namespace references have a Path to the parent namespace, but
  // the file is actually in the subdirectory for the namespace.
  if (Type == doc::InfoType::IT_namespace)
    llvm::sys::path::append(FilePath, Name);

  return llvm::sys::path::relative_path(FilePath);
}

llvm::SmallString<64>
Reference::getRelativeFilePath(const StringRef &CurrentPath) const {
  return calculateRelativeFilePath(RefType, Path, Name, CurrentPath);
}

llvm::SmallString<16> Reference::getFileBaseName() const {
  if (RefType == InfoType::IT_namespace)
    return llvm::SmallString<16>("index");

  return Name;
}

llvm::SmallString<64>
Info::getRelativeFilePath(const StringRef &CurrentPath) const {
  return calculateRelativeFilePath(IT, Path, extractName(), CurrentPath);
}

llvm::SmallString<16> Info::getFileBaseName() const {
  if (IT == InfoType::IT_namespace)
    return llvm::SmallString<16>("index");

  return extractName();
}

bool Reference::mergeable(const Reference &Other) {
  return RefType == Other.RefType && USR == Other.USR;
}

void Reference::merge(Reference &&Other) {
  assert(mergeable(Other));
  if (Name.empty())
    Name = Other.Name;
  if (Path.empty())
    Path = Other.Path;
  if (!IsInGlobalNamespace)
    IsInGlobalNamespace = Other.IsInGlobalNamespace;
}

void Info::mergeBase(Info &&Other) {
  assert(mergeable(Other));
  if (USR == EmptySID)
    USR = Other.USR;
  if (Name == "")
    Name = Other.Name;
  if (Path == "")
    Path = Other.Path;
  if (Namespace.empty())
    Namespace = std::move(Other.Namespace);
  // Unconditionally extend the description, since each decl may have a comment.
  std::move(Other.Description.begin(), Other.Description.end(),
            std::back_inserter(Description));
  llvm::sort(Description);
  auto Last = std::unique(Description.begin(), Description.end());
  Description.erase(Last, Description.end());
}

bool Info::mergeable(const Info &Other) {
  return IT == Other.IT && USR == Other.USR;
}

void SymbolInfo::merge(SymbolInfo &&Other) {
  assert(mergeable(Other));
  if (!DefLoc)
    DefLoc = std::move(Other.DefLoc);
  // Unconditionally extend the list of locations, since we want all of them.
  std::move(Other.Loc.begin(), Other.Loc.end(), std::back_inserter(Loc));
  llvm::sort(Loc);
  auto Last = std::unique(Loc.begin(), Loc.end());
  Loc.erase(Last, Loc.end());
  mergeBase(std::move(Other));
}

void NamespaceInfo::merge(NamespaceInfo &&Other) {
  assert(mergeable(Other));
  // Reduce children if necessary.
  reduceChildren(ChildNamespaces, std::move(Other.ChildNamespaces));
  reduceChildren(ChildRecords, std::move(Other.ChildRecords));
  reduceChildren(ChildFunctions, std::move(Other.ChildFunctions));
  reduceChildren(ChildEnums, std::move(Other.ChildEnums));
  mergeBase(std::move(Other));
}

void RecordInfo::merge(RecordInfo &&Other) {
  assert(mergeable(Other));
  if (!TagType)
    TagType = Other.TagType;
  if (Members.empty())
    Members = std::move(Other.Members);
  if (Bases.empty())
    Bases = std::move(Other.Bases);
  if (Parents.empty())
    Parents = std::move(Other.Parents);
  if (VirtualParents.empty())
    VirtualParents = std::move(Other.VirtualParents);
  // Reduce children if necessary.
  reduceChildren(ChildRecords, std::move(Other.ChildRecords));
  reduceChildren(ChildFunctions, std::move(Other.ChildFunctions));
  reduceChildren(ChildEnums, std::move(Other.ChildEnums));
  SymbolInfo::merge(std::move(Other));
}

void EnumInfo::merge(EnumInfo &&Other) {
  assert(mergeable(Other));
  if (!Scoped)
    Scoped = Other.Scoped;
  if (Members.empty())
    Members = std::move(Other.Members);
  SymbolInfo::merge(std::move(Other));
}

void FunctionInfo::merge(FunctionInfo &&Other) {
  assert(mergeable(Other));
  if (!IsMethod)
    IsMethod = Other.IsMethod;
  if (!Access)
    Access = Other.Access;
  if (ReturnType.Type.USR == EmptySID && ReturnType.Type.Name == "")
    ReturnType = std::move(Other.ReturnType);
  if (Parent.USR == EmptySID && Parent.Name == "")
    Parent = std::move(Other.Parent);
  if (Params.empty())
    Params = std::move(Other.Params);
  SymbolInfo::merge(std::move(Other));
}

llvm::SmallString<16> Info::extractName() const {
  if (!Name.empty())
    return Name;

  switch (IT) {
  case InfoType::IT_namespace:
    // Cover the case where the project contains a base namespace called
    // 'GlobalNamespace' (i.e. a namespace at the same level as the global
    // namespace, which would conflict with the hard-coded global namespace name
    // below.)
    if (Name == "GlobalNamespace" && Namespace.empty())
      return llvm::SmallString<16>("@GlobalNamespace");
    // The case of anonymous namespaces is taken care of in serialization,
    // so here we can safely assume an unnamed namespace is the global
    // one.
    return llvm::SmallString<16>("GlobalNamespace");
  case InfoType::IT_record:
    return llvm::SmallString<16>("@nonymous_record_" +
                                 toHex(llvm::toStringRef(USR)));
  case InfoType::IT_enum:
    return llvm::SmallString<16>("@nonymous_enum_" +
                                 toHex(llvm::toStringRef(USR)));
  case InfoType::IT_function:
    return llvm::SmallString<16>("@nonymous_function_" +
                                 toHex(llvm::toStringRef(USR)));
  case InfoType::IT_default:
    return llvm::SmallString<16>("@nonymous_" + toHex(llvm::toStringRef(USR)));
  }
  llvm_unreachable("Invalid InfoType.");
  return llvm::SmallString<16>("");
}

// Order is based on the Name attribute: case insensitive order
bool Index::operator<(const Index &Other) const {
  // Loop through each character of both strings
  for (unsigned I = 0; I < Name.size() && I < Other.Name.size(); ++I) {
    // Compare them after converting both to lower case
    int D = tolower(Name[I]) - tolower(Other.Name[I]);
    if (D == 0)
      continue;
    return D < 0;
  }
  // If both strings have the size it means they would be equal if changed to
  // lower case. In here, lower case will be smaller than upper case
  // Example: string < stRing = true
  // This is the opposite of how operator < handles strings
  if (Name.size() == Other.Name.size())
    return Name > Other.Name;
  // If they are not the same size; the shorter string is smaller
  return Name.size() < Other.Name.size();
}

void Index::sort() {
  llvm::sort(Children);
  for (auto &C : Children)
    C.sort();
}

ClangDocContext::ClangDocContext(tooling::ExecutionContext *ECtx,
                                 StringRef ProjectName, bool PublicOnly,
                                 StringRef OutDirectory, StringRef SourceRoot,
                                 StringRef RepositoryUrl,
                                 std::vector<std::string> UserStylesheets,
                                 std::vector<std::string> JsScripts)
    : ECtx(ECtx), ProjectName(ProjectName), PublicOnly(PublicOnly),
      OutDirectory(OutDirectory), UserStylesheets(UserStylesheets),
      JsScripts(JsScripts) {
  llvm::SmallString<128> SourceRootDir(SourceRoot);
  if (SourceRoot.empty())
    // If no SourceRoot was provided the current path is used as the default
    llvm::sys::fs::current_path(SourceRootDir);
  this->SourceRoot = std::string(SourceRootDir.str());
  if (!RepositoryUrl.empty()) {
    this->RepositoryUrl = std::string(RepositoryUrl);
    if (!RepositoryUrl.empty() && RepositoryUrl.find("http://") != 0 &&
        RepositoryUrl.find("https://") != 0)
      this->RepositoryUrl->insert(0, "https://");
  }
}

} // namespace doc
} // namespace clang
