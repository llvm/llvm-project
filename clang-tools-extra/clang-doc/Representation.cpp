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
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace doc {

// Thread local arenas usable in each thread pool
thread_local llvm::BumpPtrAllocator TransientArena;
thread_local llvm::BumpPtrAllocator PersistentArena;

ConcurrentStringPool &getGlobalStringPool() {
  static ConcurrentStringPool GlobalPool;
  return GlobalPool;
}

CommentKind stringToCommentKind(llvm::StringRef KindStr) {
  static const llvm::StringMap<CommentKind> KindMap = {
      {"FullComment", CommentKind::CK_FullComment},
      {"ParagraphComment", CommentKind::CK_ParagraphComment},
      {"TextComment", CommentKind::CK_TextComment},
      {"InlineCommandComment", CommentKind::CK_InlineCommandComment},
      {"HTMLStartTagComment", CommentKind::CK_HTMLStartTagComment},
      {"HTMLEndTagComment", CommentKind::CK_HTMLEndTagComment},
      {"BlockCommandComment", CommentKind::CK_BlockCommandComment},
      {"ParamCommandComment", CommentKind::CK_ParamCommandComment},
      {"TParamCommandComment", CommentKind::CK_TParamCommandComment},
      {"VerbatimBlockComment", CommentKind::CK_VerbatimBlockComment},
      {"VerbatimBlockLineComment", CommentKind::CK_VerbatimBlockLineComment},
      {"VerbatimLineComment", CommentKind::CK_VerbatimLineComment},
  };

  auto It = KindMap.find(KindStr);
  if (It != KindMap.end()) {
    return It->second;
  }
  return CommentKind::CK_Unknown;
}

llvm::StringRef commentKindToString(CommentKind Kind) {
  switch (Kind) {
  case CommentKind::CK_FullComment:
    return "FullComment";
  case CommentKind::CK_ParagraphComment:
    return "ParagraphComment";
  case CommentKind::CK_TextComment:
    return "TextComment";
  case CommentKind::CK_InlineCommandComment:
    return "InlineCommandComment";
  case CommentKind::CK_HTMLStartTagComment:
    return "HTMLStartTagComment";
  case CommentKind::CK_HTMLEndTagComment:
    return "HTMLEndTagComment";
  case CommentKind::CK_BlockCommandComment:
    return "BlockCommandComment";
  case CommentKind::CK_ParamCommandComment:
    return "ParamCommandComment";
  case CommentKind::CK_TParamCommandComment:
    return "TParamCommandComment";
  case CommentKind::CK_VerbatimBlockComment:
    return "VerbatimBlockComment";
  case CommentKind::CK_VerbatimBlockLineComment:
    return "VerbatimBlockLineComment";
  case CommentKind::CK_VerbatimLineComment:
    return "VerbatimLineComment";
  case CommentKind::CK_Unknown:
    return "Unknown";
  }
  llvm_unreachable("Unhandled CommentKind");
}

const SymbolID EmptySID = SymbolID();

template <typename T>
static llvm::Expected<OwnedPtr<Info>> reduce(OwningPtrArray<Info> &Values) {
  if (Values.empty() || !Values[0])
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no value to reduce");
  OwnedPtr<Info> Merged = allocatePtr<T>(Values[0]->USR);
  T *Tmp = static_cast<T *>(getPtr(Merged));
  for (auto &I : Values)
    Tmp->merge(std::move(*static_cast<T *>(getPtr(I))));
  return std::move(Merged);
}

template <typename T>
static void reduceChildren(llvm::simple_ilist<T> &Children,
                           llvm::simple_ilist<T> &&ChildrenToMerge) {
  while (!ChildrenToMerge.empty()) {
    T *ChildToMerge = &ChildrenToMerge.front();
    ChildrenToMerge.pop_front();

    auto It = llvm::find_if(
        Children, [&](const T &C) { return C.USR == ChildToMerge->USR; });
    if (It == Children.end()) {
      T *NewChild = allocatePtr<T>(PersistentArena, ChildToMerge->USR);
      NewChild->merge(std::move(*ChildToMerge));
      Children.push_back(*NewChild);
    } else {
      It->merge(std::move(*ChildToMerge));
    }
  }
}

template <>
void reduceChildren<Reference>(
    llvm::simple_ilist<Reference> &Children,
    llvm::simple_ilist<Reference> &&ChildrenToMerge) {
  while (!ChildrenToMerge.empty()) {
    Reference *ChildToMerge = &ChildrenToMerge.front();
    ChildrenToMerge.pop_front();

    auto It = llvm::find_if(Children, [&](const Reference &C) {
      return C.USR == ChildToMerge->USR;
    });
    if (It == Children.end()) {
      Reference *NewChild = allocatePtr<Reference>(PersistentArena);
      NewChild->USR = ChildToMerge->USR;
      NewChild->RefType = ChildToMerge->RefType;
      NewChild->merge(std::move(*ChildToMerge));
      Children.push_back(*NewChild);
    } else {
      It->merge(std::move(*ChildToMerge));
    }
  }
}

template <typename Container>
static void mergeUnkeyed(Container &Target, Container &&Source) {
  using T = typename Container::value_type;
  while (!Source.empty()) {
    auto &Item = Source.front();
    Source.pop_front();
    if (llvm::none_of(Target, [&](const auto &E) { return E == Item; })) {
      T *NewItem = allocatePtr<T>(PersistentArena, Item);
      Target.push_back(*NewItem);
    }
  }
}

template <>
void mergeUnkeyed<OwningVec<CommentInfo>>(OwningVec<CommentInfo> &Target,
                                          OwningVec<CommentInfo> &&Source) {
  while (!Source.empty()) {
    auto &Item = Source.front();
    Source.pop_front();
    if (llvm::none_of(Target, [&](const auto &E) { return E == Item; })) {
      CommentInfo *NewItem =
          allocatePtr<CommentInfo>(PersistentArena, Item, PersistentArena);
      Target.push_back(*NewItem);
    }
  }
}

llvm::Error mergeSingleInfo(doc::OwnedPtr<doc::Info> &Reduced,
                            doc::OwnedPtr<doc::Info> &&NewInfo,
                            llvm::BumpPtrAllocator &Arena) {
  if (!Reduced) {
    switch (NewInfo->IT) {
    case InfoType::IT_namespace:
      Reduced = allocatePtr<NamespaceInfo>(Arena, NewInfo->USR);
      break;
    case InfoType::IT_record:
      Reduced = allocatePtr<RecordInfo>(Arena, NewInfo->USR);
      break;
    case InfoType::IT_enum:
      Reduced = allocatePtr<EnumInfo>(Arena, NewInfo->USR);
      break;
    case InfoType::IT_function:
      Reduced = allocatePtr<FunctionInfo>(Arena, NewInfo->USR);
      break;
    case InfoType::IT_typedef:
      Reduced = allocatePtr<TypedefInfo>(Arena, NewInfo->USR);
      break;
    case InfoType::IT_concept:
      Reduced = allocatePtr<ConceptInfo>(Arena, NewInfo->USR);
      break;
    case InfoType::IT_variable:
      Reduced = allocatePtr<VarInfo>(Arena, NewInfo->USR);
      break;
    case InfoType::IT_friend:
      Reduced = allocatePtr<FriendInfo>(Arena, NewInfo->USR);
      break;
    default:
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unknown info type");
    }
  }

  if (Reduced->IT != NewInfo->IT)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "info types mismatch");

  switch (Reduced->IT) {
  case InfoType::IT_namespace:
    static_cast<NamespaceInfo *>(getPtr(Reduced))
        ->merge(std::move(*static_cast<NamespaceInfo *>(getPtr(NewInfo))));
    break;
  case InfoType::IT_record:
    static_cast<RecordInfo *>(getPtr(Reduced))
        ->merge(std::move(*static_cast<RecordInfo *>(getPtr(NewInfo))));
    break;
  case InfoType::IT_enum:
    static_cast<EnumInfo *>(getPtr(Reduced))
        ->merge(std::move(*static_cast<EnumInfo *>(getPtr(NewInfo))));
    break;
  case InfoType::IT_function:
    static_cast<FunctionInfo *>(getPtr(Reduced))
        ->merge(std::move(*static_cast<FunctionInfo *>(getPtr(NewInfo))));
    break;
  case InfoType::IT_typedef:
    static_cast<TypedefInfo *>(getPtr(Reduced))
        ->merge(std::move(*static_cast<TypedefInfo *>(getPtr(NewInfo))));
    break;
  case InfoType::IT_concept:
    static_cast<ConceptInfo *>(getPtr(Reduced))
        ->merge(std::move(*static_cast<ConceptInfo *>(getPtr(NewInfo))));
    break;
  case InfoType::IT_variable:
    static_cast<VarInfo *>(getPtr(Reduced))
        ->merge(std::move(*static_cast<VarInfo *>(getPtr(NewInfo))));
    break;
  case InfoType::IT_friend:
    static_cast<FriendInfo *>(getPtr(Reduced))
        ->merge(std::move(*static_cast<FriendInfo *>(getPtr(NewInfo))));
    break;
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unknown info type");
  }

  return llvm::Error::success();
}

// Dispatch function.
llvm::Expected<OwnedPtr<Info>> mergeInfos(OwningPtrArray<Info> &Values) {
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
  case InfoType::IT_typedef:
    return reduce<TypedefInfo>(Values);
  case InfoType::IT_concept:
    return reduce<ConceptInfo>(Values);
  case InfoType::IT_variable:
    return reduce<VarInfo>(Values);
  case InfoType::IT_friend:
    return reduce<FriendInfo>(Values);
  case InfoType::IT_default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unexpected info type");
  }
  llvm_unreachable("unhandled enumerator");
}

TemplateSpecializationInfo::TemplateSpecializationInfo(
    const TemplateSpecializationInfo &Other, llvm::BumpPtrAllocator &Arena)
    : SpecializationOf(Other.SpecializationOf) {
  Params = allocateArray(Other.Params, Arena);
}

TemplateInfo::TemplateInfo(const TemplateInfo &Other,
                           llvm::BumpPtrAllocator &Arena) {
  Params = allocateArray(Other.Params, Arena);
  if (Other.Specialization)
    Specialization = TemplateSpecializationInfo(*Other.Specialization, Arena);
  Constraints = allocateArray(Other.Constraints, Arena);
}

bool CommentInfo::operator==(const CommentInfo &Other) const {
  auto FirstCI = std::tie(Kind, Text, Name, Direction, ParamName, CloseName,
                          SelfClosing, Explicit, AttrKeys, AttrValues, Args);
  auto SecondCI =
      std::tie(Other.Kind, Other.Text, Other.Name, Other.Direction,
               Other.ParamName, Other.CloseName, Other.SelfClosing,
               Other.Explicit, Other.AttrKeys, Other.AttrValues, Other.Args);

  if (FirstCI != SecondCI || Children.size() != Other.Children.size())
    return false;

  return std::equal(Children.begin(), Children.end(), Other.Children.begin(),
                    Other.Children.end());
}

bool CommentInfo::operator<(const CommentInfo &Other) const {
  auto FirstCI = std::tie(Kind, Text, Name, Direction, ParamName, CloseName,
                          SelfClosing, Explicit, AttrKeys, AttrValues, Args);
  auto SecondCI =
      std::tie(Other.Kind, Other.Text, Other.Name, Other.Direction,
               Other.ParamName, Other.CloseName, Other.SelfClosing,
               Other.Explicit, Other.AttrKeys, Other.AttrValues, Other.Args);

  if (FirstCI < SecondCI)
    return true;

  if (FirstCI == SecondCI) {
    return std::lexicographical_compare(Children.begin(), Children.end(),
                                        Other.Children.begin(),
                                        Other.Children.end());
  }

  return false;
}

CommentInfo::CommentInfo(const CommentInfo &Other,
                         llvm::BumpPtrAllocator &Arena) {
  Kind = Other.Kind;
  Direction = Other.Direction;
  Name = Other.Name;
  ParamName = Other.ParamName;
  CloseName = Other.CloseName;
  SelfClosing = Other.SelfClosing;
  Explicit = Other.Explicit;
  Text = Other.Text;
  AttrKeys = allocateArray(Other.AttrKeys, Arena);
  AttrValues = allocateArray(Other.AttrValues, Arena);
  Args = allocateArray(Other.Args, Arena);
  if (!Other.Children.empty()) {
    CommentInfo *NewArray = Arena.Allocate<CommentInfo>(Other.Children.size());
    for (size_t Idx = 0; Idx < Other.Children.size(); ++Idx) {
      new (NewArray + Idx) CommentInfo(Other.Children[Idx], Arena);
    }
    Children = llvm::ArrayRef<CommentInfo>(NewArray, Other.Children.size());
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

StringRef Reference::getRelativeFilePath(const StringRef &CurrentPath) const {
  return internString(
      calculateRelativeFilePath(RefType, Path, Name, CurrentPath));
}

StringRef Reference::getFileBaseName() const {
  if (RefType == InfoType::IT_namespace)
    return "index";

  return Name;
}

StringRef Info::getRelativeFilePath(const StringRef &CurrentPath) const {
  return internString(
      calculateRelativeFilePath(IT, Path, extractName(), CurrentPath));
}

StringRef Info::getFileBaseName() const {
  if (IT == InfoType::IT_namespace)
    return "index";

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
  if (QualName.empty())
    QualName = Other.QualName;
  if (DocumentationFileName.empty())
    DocumentationFileName = Other.DocumentationFileName;
}

bool FriendInfo::mergeable(const FriendInfo &Other) {
  return Ref.USR == Other.Ref.USR && Ref.Name == Other.Ref.Name;
}

void FriendInfo::merge(FriendInfo &&Other) {
  assert(mergeable(Other));
  Ref.merge(std::move(Other.Ref));
  SymbolInfo::merge(std::move(Other));
}

FriendInfo::FriendInfo(const FriendInfo &Other, llvm::BumpPtrAllocator &Arena)
    : SymbolInfo(Other, Arena) {
  Ref = Other.Ref;
  if (Other.Template)
    Template.emplace(*Other.Template, Arena);
  if (Other.ReturnType)
    ReturnType = Other.ReturnType;
  if (!Other.Params.empty())
    Params = allocateArray(Other.Params, Arena);
  IsClass = Other.IsClass;
}

Info::Info(const Info &Other, llvm::BumpPtrAllocator &Arena)
    : Path(Other.Path), Name(Other.Name),
      DocumentationFileName(Other.DocumentationFileName), USR(Other.USR),
      ParentUSR(Other.ParentUSR), IT(Other.IT) {
  Namespace = allocateArray(Other.Namespace, Arena);
  if (!Other.Description.empty()) {
    for (const auto &Desc : Other.Description) {
      CommentInfo *NewDesc = allocatePtr<CommentInfo>(Arena, Desc, Arena);
      Description.push_back(*NewDesc);
    }
  }
}

void Info::mergeBase(Info &&Other) {
  assert(mergeable(Other));
  if (USR == EmptySID)
    USR = Other.USR;
  if (Name == "")
    Name = Other.Name;
  if (Path == "")
    Path = Other.Path;
  if (Namespace.empty() && !Other.Namespace.empty())
    Namespace = allocateArray(Other.Namespace, PersistentArena);
  // Unconditionally extend the description, since each decl may have a comment.
  mergeUnkeyed(Description, std::move(Other.Description));
  if (ParentUSR == EmptySID)
    ParentUSR = Other.ParentUSR;
  if (DocumentationFileName.empty())
    DocumentationFileName = Other.DocumentationFileName;
}

bool Info::mergeable(const Info &Other) {
  return IT == Other.IT && USR == Other.USR;
}

SymbolInfo::SymbolInfo(const SymbolInfo &Other, llvm::BumpPtrAllocator &Arena)
    : Info(Other, Arena), DefLoc(Other.DefLoc), MangledName(Other.MangledName),
      IsStatic(Other.IsStatic) {
  if (!Other.Loc.empty()) {
    for (const auto &L : Other.Loc) {
      Location *NewL = allocatePtr<Location>(Arena, L);
      Loc.push_back(*NewL);
    }
  }
}

void SymbolInfo::merge(SymbolInfo &&Other) {
  assert(mergeable(Other));
  if (!DefLoc)
    DefLoc = std::move(Other.DefLoc);
  // Unconditionally extend the list of locations, since we want all of them.
  mergeUnkeyed(Loc, std::move(Other.Loc));
  mergeBase(std::move(Other));
  if (MangledName.empty())
    MangledName = std::move(Other.MangledName);
  if (!IsStatic)
    IsStatic = Other.IsStatic;
}

NamespaceInfo::NamespaceInfo(SymbolID USR, StringRef Name, StringRef Path)
    : Info(InfoType::IT_namespace, USR, Name, Path) {}

void NamespaceInfo::merge(NamespaceInfo &&Other) {
  assert(mergeable(Other));
  // Reduce children if necessary.
  reduceChildren(Children.Namespaces, std::move(Other.Children.Namespaces));
  reduceChildren(Children.Records, std::move(Other.Children.Records));
  reduceChildren(Children.Functions, std::move(Other.Children.Functions));
  reduceChildren(Children.Enums, std::move(Other.Children.Enums));
  reduceChildren(Children.Typedefs, std::move(Other.Children.Typedefs));
  reduceChildren(Children.Concepts, std::move(Other.Children.Concepts));
  reduceChildren(Children.Variables, std::move(Other.Children.Variables));
  mergeBase(std::move(Other));
}

RecordInfo::RecordInfo(SymbolID USR, StringRef Name, StringRef Path)
    : SymbolInfo(InfoType::IT_record, USR, Name, Path) {}

// FIXME: This constructor is currently unsafe for cross-arena copies of
// populated records. Because a default copy of ScopeChildren will shallow-copy
// the intrusive pointers, leading to a use-after-free when the TransientArena
// is reset. Subsequent patches will address this by deep-copying children
// individually via reduceChildren.
RecordInfo::RecordInfo(const RecordInfo &Other, llvm::BumpPtrAllocator &Arena)
    : SymbolInfo(Other, Arena), TagType(Other.TagType),
      IsTypeDef(Other.IsTypeDef) {
  Members = deepCopyArray(Other.Members, Arena);
  Parents = allocateArray(Other.Parents, Arena);
  VirtualParents = allocateArray(Other.VirtualParents, Arena);
  Bases = deepCopyArray(Other.Bases, Arena);
  Friends = deepCopyArray(Other.Friends, Arena);
}

MemberTypeInfo::MemberTypeInfo(const MemberTypeInfo &Other,
                               llvm::BumpPtrAllocator &Arena)
    : FieldTypeInfo(Other), Access(Other.Access), IsStatic(Other.IsStatic) {
  if (!Other.Description.empty()) {
    for (const auto &Desc : Other.Description) {
      CommentInfo *NewDesc = allocatePtr<CommentInfo>(Arena, Desc, Arena);
      Description.push_back(*NewDesc);
    }
  }
}

void RecordInfo::merge(RecordInfo &&Other) {
  assert(mergeable(Other));
  if (!llvm::to_underlying(TagType))
    TagType = Other.TagType;
  IsTypeDef = IsTypeDef || Other.IsTypeDef;
  if (Members.empty() && !Other.Members.empty())
    Members = deepCopyArray(Other.Members, PersistentArena);
  if (Bases.empty() && !Other.Bases.empty())
    Bases = deepCopyArray(Other.Bases, PersistentArena);
  if (Parents.empty() && !Other.Parents.empty())
    Parents = allocateArray(Other.Parents, PersistentArena);
  if (VirtualParents.empty() && !Other.VirtualParents.empty())
    VirtualParents = allocateArray(Other.VirtualParents, PersistentArena);
  if (Friends.empty() && !Other.Friends.empty())
    Friends = deepCopyArray(Other.Friends, PersistentArena);
  // Reduce children if necessary.
  reduceChildren(Children.Records, std::move(Other.Children.Records));
  reduceChildren(Children.Functions, std::move(Other.Children.Functions));
  reduceChildren(Children.Enums, std::move(Other.Children.Enums));
  reduceChildren(Children.Typedefs, std::move(Other.Children.Typedefs));
  SymbolInfo::merge(std::move(Other));
  if (!Template && Other.Template)
    Template = TemplateInfo(*Other.Template, PersistentArena);
}

EnumValueInfo::EnumValueInfo(const EnumValueInfo &Other,
                             llvm::BumpPtrAllocator &Arena)
    : Name(Other.Name), Value(Other.Value), ValueExpr(Other.ValueExpr) {
  if (!Other.Description.empty()) {
    for (const auto &Desc : Other.Description) {
      CommentInfo *NewDesc = allocatePtr<CommentInfo>(Arena, Desc, Arena);
      Description.push_back(*NewDesc);
    }
  }
}

void EnumInfo::merge(EnumInfo &&Other) {
  assert(mergeable(Other));
  if (!Scoped)
    Scoped = Other.Scoped;
  if (!BaseType && Other.BaseType)
    BaseType = std::move(Other.BaseType);
  if (Members.empty() && !Other.Members.empty())
    Members = deepCopyArray(Other.Members, PersistentArena);
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
  if (Params.empty() && !Other.Params.empty())
    Params = allocateArray(Other.Params, PersistentArena);
  SymbolInfo::merge(std::move(Other));
  if (!Template && Other.Template)
    Template = TemplateInfo(*Other.Template, PersistentArena);
}

void TypedefInfo::merge(TypedefInfo &&Other) {
  assert(mergeable(Other));
  if (!IsUsing)
    IsUsing = Other.IsUsing;
  if (Underlying.Type.Name == "")
    Underlying = Other.Underlying;
  if (!Template && Other.Template)
    Template = TemplateInfo(*Other.Template, PersistentArena);
  SymbolInfo::merge(std::move(Other));
}

void ConceptInfo::merge(ConceptInfo &&Other) {
  assert(mergeable(Other));
  if (!IsType)
    IsType = Other.IsType;
  if (ConstraintExpression.empty())
    ConstraintExpression = std::move(Other.ConstraintExpression);
  if (Template.Constraints.empty() && !Other.Template.Constraints.empty())
    Template.Constraints =
        allocateArray(Other.Template.Constraints, PersistentArena);
  if (Template.Params.empty() && !Other.Template.Params.empty())
    Template.Params = allocateArray(Other.Template.Params, PersistentArena);
  SymbolInfo::merge(std::move(Other));
}

void VarInfo::merge(VarInfo &&Other) {
  assert(mergeable(Other));
  if (!IsStatic)
    IsStatic = Other.IsStatic;
  if (Type.Type.USR == EmptySID && Type.Type.Name == "")
    Type = std::move(Other.Type);
  SymbolInfo::merge(std::move(Other));
}

BaseRecordInfo::BaseRecordInfo() : RecordInfo() {}

BaseRecordInfo::BaseRecordInfo(const BaseRecordInfo &Other,
                               llvm::BumpPtrAllocator &Arena)
    : RecordInfo(Other, Arena), Access(Other.Access),
      IsVirtual(Other.IsVirtual), IsParent(Other.IsParent) {}

BaseRecordInfo::BaseRecordInfo(SymbolID USR, StringRef Name, StringRef Path,
                               bool IsVirtual, AccessSpecifier Access,
                               bool IsParent)
    : RecordInfo(USR, Name, Path), Access(Access), IsVirtual(IsVirtual),
      IsParent(IsParent) {}

StringRef Info::extractName() const {
  if (!Name.empty())
    return Name;

  switch (IT) {
  case InfoType::IT_namespace:
    // Cover the case where the project contains a base namespace called
    // 'GlobalNamespace' (i.e. a namespace at the same level as the global
    // namespace, which would conflict with the hard-coded global namespace name
    // below.)
    if (Name == "GlobalNamespace" && Namespace.empty())
      return "@GlobalNamespace";
    // The case of anonymous namespaces is taken care of in serialization,
    // so here we can safely assume an unnamed namespace is the global
    // one.
    return "GlobalNamespace";
  case InfoType::IT_record:
    return internString("@nonymous_record_" + toHex(llvm::toStringRef(USR)));
  case InfoType::IT_enum:
    return internString("@nonymous_enum_" + toHex(llvm::toStringRef(USR)));
  case InfoType::IT_typedef:
    return internString("@nonymous_typedef_" + toHex(llvm::toStringRef(USR)));
  case InfoType::IT_function:
    return internString("@nonymous_function_" + toHex(llvm::toStringRef(USR)));
  case InfoType::IT_concept:
    return internString("@nonymous_concept_" + toHex(llvm::toStringRef(USR)));
  case InfoType::IT_variable:
    return internString("@nonymous_variable_" + toHex(llvm::toStringRef(USR)));
  case InfoType::IT_friend:
    return internString("@nonymous_friend_" + toHex(llvm::toStringRef(USR)));
  case InfoType::IT_default:
    return internString("@nonymous_" + toHex(llvm::toStringRef(USR)));
  }
  llvm_unreachable("Invalid InfoType.");
  return "";
}

// Order is based on the Name attribute: case insensitive order
bool Index::operator<(const Index &Other) const {
  // Start with case-insensitive (e.g., 'apple' < 'Zebra').
  // This prevents 'Zebra' from appearing before 'apple' due to ASCII values,
  // where uppercase letters have a lower numeric value than lowercase.
  int Cmp = Name.compare_insensitive(Other.Name);
  if (Cmp != 0)
    return Cmp < 0;

  // If names are identical, we fall back to standard string comparison where
  // uppercase precedes lowercase (e.g., 'Apple' < 'apple').
  return Name < Other.Name;
}

std::vector<const Index *> Index::getSortedChildren() const {
  std::vector<const Index *> SortedChildren;
  SortedChildren.reserve(Children.size());
  for (const auto &[_, C] : Children)
    SortedChildren.push_back(&C);
  llvm::sort(SortedChildren,
             [](const Index *A, const Index *B) { return *A < *B; });
  return SortedChildren;
}

void Index::sort() {
  for (auto &[_, C] : Children)
    C.sort();
}

ClangDocContext::ClangDocContext(tooling::ExecutionContext *ECtx,
                                 StringRef ProjectName, bool PublicOnly,
                                 StringRef OutDirectory, StringRef SourceRoot,
                                 StringRef RepositoryUrl,
                                 StringRef RepositoryLinePrefix, StringRef Base,
                                 std::vector<std::string> UserStylesheets,
                                 clang::DiagnosticsEngine &Diags,
                                 OutputFormatTy Format, bool FTimeTrace)
    : ECtx(ECtx), ProjectName(ProjectName), OutDirectory(OutDirectory),
      SourceRoot(std::string(SourceRoot)), UserStylesheets(UserStylesheets),
      Base(Base), Diags(Diags), Format(Format), PublicOnly(PublicOnly),
      FTimeTrace(FTimeTrace) {
  llvm::SmallString<128> SourceRootDir(SourceRoot);
  if (SourceRoot.empty())
    // If no SourceRoot was provided the current path is used as the default
    llvm::sys::fs::current_path(SourceRootDir);
  this->SourceRoot = std::string(SourceRootDir);
  if (!RepositoryUrl.empty()) {
    this->RepositoryUrl = std::string(RepositoryUrl);
    if (!RepositoryUrl.empty() && !RepositoryUrl.starts_with("http://") &&
        !RepositoryUrl.starts_with("https://"))
      this->RepositoryUrl->insert(0, "https://");

    if (!RepositoryLinePrefix.empty())
      this->RepositoryLinePrefix = std::string(RepositoryLinePrefix);
  }
}

void ScopeChildren::sort() {
  Namespaces.sort();
  Records.sort();
  Functions.sort();
  Enums.sort();
  Typedefs.sort();
  Concepts.sort();
  Variables.sort();
}
} // namespace doc
} // namespace clang
