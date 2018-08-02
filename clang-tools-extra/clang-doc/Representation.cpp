///===-- Representation.cpp - ClangDoc Representation -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

static const SymbolID EmptySID = SymbolID();

template <typename T>
std::unique_ptr<Info> reduce(std::vector<std::unique_ptr<Info>> &Values) {
  std::unique_ptr<Info> Merged = llvm::make_unique<T>();
  T *Tmp = static_cast<T *>(Merged.get());
  for (auto &I : Values)
    Tmp->merge(std::move(*static_cast<T *>(I.get())));
  return Merged;
}

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
  return IT == Other.IT && (USR == EmptySID || USR == Other.USR);
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
