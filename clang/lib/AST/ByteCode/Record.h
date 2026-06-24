//===--- Record.h - struct and class metadata for the VM --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A record is part of a program to describe the layout and methods of a struct.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_RECORD_H
#define LLVM_CLANG_AST_INTERP_RECORD_H

#include "PrimType.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

namespace clang {
namespace interp {
class Program;
struct Descriptor;

/// Structure/Class descriptor.
class Record final {
public:
  /// Describes a record field.
  struct Field {
    const FieldDecl *Decl;
    const Descriptor *Desc;
    unsigned Offset;

    bool isBitField() const { return Decl->isBitField(); }
    bool isUnnamedBitField() const { return Decl->isUnnamedBitField(); }
    unsigned bitWidth() const {
      assert(isBitField());
      return Decl->getBitWidthValue();
    }

    Field(const FieldDecl *D, const Descriptor *Desc, unsigned Offset)
        : Decl(D), Desc(Desc), Offset(Offset) {}
  };

  /// Describes a base class.
  struct Base {
    const RecordDecl *Decl;
    const Descriptor *Desc;
    const Record *R;
    unsigned Offset;

    Base(const RecordDecl *D, const Descriptor *Desc, const Record *R,
         unsigned Offset)
        : Decl(D), Desc(Desc), R(R), Offset(Offset) {}
  };

public:
  static size_t allocSize(unsigned NumFields, unsigned NumBases,
                          unsigned NumVBases) {
    return align(sizeof(Record)) + align((NumFields * sizeof(Field))) +
           align((NumBases * sizeof(Base))) + align(NumVBases * sizeof(Base));
  }

  /// Returns the underlying declaration.
  const RecordDecl *getDecl() const { return Decl; }
  /// Returns the name of the underlying declaration.
  std::string getName() const;
  /// Checks if the record is a union.
  bool isUnion() const { return IsUnion; }
  /// Checks if the record is an anonymous union.
  bool isAnonymousUnion() const { return IsAnonymousUnion; }
  /// Returns the size of the record.
  unsigned getSize() const { return BaseSize; }
  /// Returns the full size of the record, including records.
  unsigned getFullSize() const { return BaseSize + VirtualSize; }
  /// Returns the destructor of the record, if any.
  const CXXDestructorDecl *getDestructor() const {
    if (const auto *CXXDecl = dyn_cast<CXXRecordDecl>(Decl))
      return CXXDecl->getDestructor();
    return nullptr;
  }
  /// If this record (or any of its bases) contains a field of type PT_Ptr.
  bool hasPtrField() const { return HasPtrField; }

  /// Returns true for anonymous unions and records
  /// with no destructor or for those with a trivial destructor.
  bool hasTrivialDtor() const;

  using const_field_iter = ArrayRef<Field>::const_iterator;
  llvm::iterator_range<const_field_iter> fields() const {
    return llvm::make_range(getFields(), getFields() + NumFields);
  }

  unsigned getNumFields() const { return NumFields; }
  const Field *getField(unsigned I) const { return &getFields()[I]; }
  /// Returns a field.
  const Field *getField(const FieldDecl *FD) const {
    return &getFields()[FD->getFieldIndex()];
  }

  using const_base_iter = ArrayRef<Base>::const_iterator;
  llvm::iterator_range<const_base_iter> bases() const {
    return llvm::make_range(getBases(), getBases() + NumBases);
  }

  unsigned getNumBases() const { return NumBases; }
  const Base *getBase(unsigned I) const {
    assert(I < getNumBases());
    return &getBases()[I];
  }
  /// Returns a base descriptor.
  const Base *getBase(QualType T) const;
  /// Returns a base descriptor.
  const Base *getBase(const RecordDecl *FD) const;

  using const_vbase_iter = ArrayRef<Base>::const_iterator;
  llvm::iterator_range<const_vbase_iter> virtual_bases() const {
    return llvm::make_range(getVBases(), getVBases() + NumVBases);
  }

  unsigned getNumVirtualBases() const { return NumVBases; }
  const Base *getVirtualBase(unsigned I) const { return &getVBases()[I]; }
  /// Returns a virtual base descriptor.
  const Base *getVirtualBase(const RecordDecl *RD) const;

  void dump(llvm::raw_ostream &OS, unsigned Indentation = 0,
            unsigned Offset = 0) const;
  void dump() const { dump(llvm::errs()); }

private:
  /// Constructor used by Program to create record descriptors.
  Record(const RecordDecl *, unsigned NumBases, unsigned NumFields,
         unsigned NumVBases);

private:
  friend class Program;

  Field *getFields() const {
    return reinterpret_cast<Field *>(
        (reinterpret_cast<char *>(const_cast<Record *>(this))) +
        align(sizeof(*this)));
  }
  Base *getBases() const {
    return reinterpret_cast<Base *>(
        (reinterpret_cast<char *>(const_cast<Record *>(this))) +
        align(sizeof(*this)) + align((NumFields * sizeof(Field))));
  }
  Base *getVBases() const {
    return reinterpret_cast<Base *>(
        (reinterpret_cast<char *>(const_cast<Record *>(this))) +
        align(sizeof(*this)) + align((NumFields * sizeof(Field))) +
        align(NumBases * sizeof(Base)));
  }

  /// Original declaration.
  const RecordDecl *Decl;
  unsigned NumBases;
  const unsigned NumFields;
  unsigned NumVBases;

  /// Mapping from declarations to bases.
  llvm::DenseMap<const RecordDecl *, const Base *> BaseMap;
  /// Mapping from declarations to virtual bases.
  llvm::DenseMap<const RecordDecl *, Base *> VirtualBaseMap;
  /// Size of the structure.
  unsigned BaseSize;
  /// Size of all virtual bases.
  unsigned VirtualSize;
  /// If this record is a union.
  bool IsUnion;
  /// If this is an anonymous union.
  bool IsAnonymousUnion;
  /// If any of the fields are pointers (or references).
  bool HasPtrField = false;
};

} // namespace interp
} // namespace clang

#endif
