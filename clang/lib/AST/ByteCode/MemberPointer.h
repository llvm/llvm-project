//===------------------------- MemberPointer.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_MEMBER_POINTER_H
#define LLVM_CLANG_AST_INTERP_MEMBER_POINTER_H

#include "Pointer.h"
#include "llvm/ADT/PointerIntPair.h"
#include <optional>

namespace clang {
class ASTContext;
class CXXRecordDecl;
namespace interp {

class Context;
class FunctionPointer;

class MemberPointer final {
private:
  Pointer Base;
  /// The member declaration, and a flag indicating
  /// whether the member is a member of some class derived from the class type
  /// of the member pointer.
  llvm::PointerIntPair<const ValueDecl *, 1, bool> DeclAndIsDerivedMember;
  /// The path of base/derived classes from the member declaration's
  /// class (exclusive) to the class type of the member pointer (inclusive).
  /// This a allocated by the InterpState or the Program.
  const CXXRecordDecl **Path = nullptr;
  int32_t PtrOffset = 0;
  uint8_t PathLength = 0;

  MemberPointer(Pointer Base, const ValueDecl *Dcl, int32_t PtrOffset,
                uint8_t PathLength = 0, const CXXRecordDecl **Path = nullptr,
                bool IsDerived = false)
      : Base(Base), DeclAndIsDerivedMember(Dcl, IsDerived), Path(Path),
        PtrOffset(PtrOffset), PathLength(PathLength) {}

public:
  MemberPointer() = default;
  MemberPointer(Pointer Base, const ValueDecl *Dcl)
      : Base(Base), DeclAndIsDerivedMember(Dcl) {}
  MemberPointer(uint32_t Address, const Descriptor *D) {
    // We only reach this for Address == 0, when creating a null member pointer.
    assert(Address == 0);
  }

  MemberPointer(const ValueDecl *D) : DeclAndIsDerivedMember(D) {
    assert((isa<FieldDecl, IndirectFieldDecl, CXXMethodDecl>(D)));
  }

  uint64_t getIntegerRepresentation() const {
    assert(
        false &&
        "getIntegerRepresentation() shouldn't be reachable for MemberPointers");
    return 17;
  }

  bool hasDecl() const { return DeclAndIsDerivedMember.getPointer(); }
  bool isDerivedMember() const { return DeclAndIsDerivedMember.getInt(); }
  const ValueDecl *getDecl() const {
    return DeclAndIsDerivedMember.getPointer();
  }
  bool hasPath() const { return PathLength != 0; }
  unsigned getPathLength() const { return PathLength; }
  const CXXRecordDecl *getPathEntry(unsigned Index) const {
    return Path[Index];
  }
  const CXXRecordDecl **path() const { return Path; }
  void takePath(const CXXRecordDecl **NewPath) {
    assert(Path != NewPath);
    Path = NewPath;
  }

  // Pretend we always have a path.
  bool singleWord() const { return false; }

  std::optional<Pointer> toPointer(const Context &Ctx) const;

  FunctionPointer toFunctionPointer(const Context &Ctx) const;

  bool isBaseCastPossible() const {
    if (PtrOffset < 0)
      return true;
    return static_cast<uint64_t>(PtrOffset) <= Base.getByteOffset();
  }

  Pointer getBase() const {
    if (PtrOffset < 0)
      return Base.atField(-PtrOffset);
    return Base.atFieldSub(PtrOffset);
  }
  bool isMemberFunctionPointer() const {
    return isa_and_nonnull<CXXMethodDecl>(DeclAndIsDerivedMember.getPointer());
  }
  const CXXMethodDecl *getMemberFunction() const {
    return dyn_cast_if_present<CXXMethodDecl>(
        DeclAndIsDerivedMember.getPointer());
  }
  const FieldDecl *getField() const {
    return dyn_cast_if_present<FieldDecl>(DeclAndIsDerivedMember.getPointer());
  }

  const CXXRecordDecl *getRecordDecl() const {
    if (const FieldDecl *FD = getField())
      return cast<CXXRecordDecl>(FD->getParent());

    if (const CXXMethodDecl *MD = getMemberFunction())
      return MD->getParent();
    return nullptr;
  }

  MemberPointer atInstanceBase(unsigned Offset, uint8_t PathLength = 0,
                               const CXXRecordDecl **Path = nullptr,
                               bool NewIsDerived = false) const {
    if (Base.isZero())
      return MemberPointer(Base, DeclAndIsDerivedMember.getPointer(), Offset,
                           PathLength, Path, NewIsDerived);
    return MemberPointer(this->Base, DeclAndIsDerivedMember.getPointer(),
                         Offset + PtrOffset, PathLength, Path, NewIsDerived);
  }

  MemberPointer takeInstance(Pointer Instance) const {
    assert(this->Base.isZero());
    return MemberPointer(Instance, DeclAndIsDerivedMember.getPointer(),
                         this->PtrOffset);
  }

  APValue toAPValue(const ASTContext &) const;

  bool isZero() const { return Base.isZero() && !hasDecl(); }
  bool hasBase() const { return !Base.isZero(); }
  bool isWeak() const {
    if (const auto *MF = getMemberFunction())
      return MF->isWeak();
    return false;
  }

  void print(llvm::raw_ostream &OS) const {
    OS << "MemberPtr(" << Base << " " << (const void *)getDecl() << " + "
       << PtrOffset << ". PathLength: " << getPathLength()
       << ". IsDerived: " << isDerivedMember() << ")";
  }

  std::string toDiagnosticString(const ASTContext &Ctx) const {
    return toAPValue(Ctx).getAsString(Ctx, getDecl()->getType());
  }

  ComparisonCategoryResult compare(const MemberPointer &RHS) const {
    if (this->getDecl() == RHS.getDecl()) {

      if (this->PathLength != RHS.PathLength)
        return ComparisonCategoryResult::Unordered;

      if (PathLength != 0 &&
          std::memcmp(Path, RHS.Path, PathLength * sizeof(CXXRecordDecl *)) !=
              0)
        return ComparisonCategoryResult::Unordered;

      return ComparisonCategoryResult::Equal;
    }
    return ComparisonCategoryResult::Unordered;
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const MemberPointer &FP) {
  FP.print(OS);
  return OS;
}

} // namespace interp
} // namespace clang

#endif
