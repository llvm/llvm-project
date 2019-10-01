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

namespace clang {
namespace doc {

namespace {

const SymbolID EmptySID = SymbolID();

template <typename T>
llvm::Expected<std::unique_ptr<Info>>
reduce(std::vector<std::unique_ptr<Info>> &Values) {
  if (Values.empty())
    return llvm::make_error<llvm::StringError>(" No values to reduce.\n",
                                               llvm::inconvertibleErrorCode());
  std::unique_ptr<Info> Merged = llvm::make_unique<T>(Values[0]->USR);
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

// For References, we don't need to actually merge them, we just don't want
// duplicates.
void reduceChildren(std::vector<Reference> &Children,
                    std::vector<Reference> &&ChildrenToMerge) {
  for (auto &ChildToMerge : ChildrenToMerge) {
    if (getChildIndexIfExists(Children, ChildToMerge) == -1)
      Children.push_back(std::move(ChildToMerge));
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
  if (Values.empty())
    return llvm::make_error<llvm::StringError>("No info values to merge.\n",
                                               llvm::inconvertibleErrorCode());

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
    return llvm::make_error<llvm::StringError>("Unexpected info type.\n",
                                               llvm::inconvertibleErrorCode());
  }
}

void Info::mergeBase(Info &&Other) {
  assert(mergeable(Other));
  if (USR == EmptySID)
    USR = Other.USR;
  if (Name == "")
    Name = Other.Name;
  if (Namespace.empty())
    Namespace = std::move(Other.Namespace);
  // Unconditionally extend the description, since each decl may have a comment.
  std::move(Other.Description.begin(), Other.Description.end(),
            std::back_inserter(Description));
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

} // namespace doc
} // namespace clang
