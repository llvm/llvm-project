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

#include "Descriptor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"

namespace clang {
namespace interp {
class Program;

/// Structure/Class descriptor.
class Record final {
public:
  /// Describes a record field.
  struct Field {
    const FieldDecl *Decl;
    unsigned Offset;
    const Descriptor *Desc;
    bool isBitField() const { return Decl->isBitField(); }
    bool isUnnamedBitField() const { return Decl->isUnnamedBitField(); }
  };

  /// Describes a base class.
  struct Base {
    const RecordDecl *Decl;
    unsigned Offset;
    const Descriptor *Desc;
    const Record *R;
  };

public:
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
  /// Returns a field.
  const Field *getField(const FieldDecl *FD) const;
  /// Returns a base descriptor.
  const Base *getBase(const RecordDecl *FD) const;
  /// Returns a base descriptor.
  const Base *getBase(QualType T) const;
  /// Returns a virtual base descriptor.
  const Base *getVirtualBase(const RecordDecl *RD) const;
  /// Returns the destructor of the record, if any.
  const CXXDestructorDecl *getDestructor() const {
    if (const auto *CXXDecl = dyn_cast<CXXRecordDecl>(Decl))
      return CXXDecl->getDestructor();
    return nullptr;
  }

  using const_field_iter = ArrayRef<Field>::const_iterator;
  llvm::iterator_range<const_field_iter> fields() const {
    return llvm::make_range(Fields, Fields + NumFields);
  }

  unsigned getNumFields() const { return NumFields; }
  const Field *getField(unsigned I) const { return &Fields[I]; }

  using const_base_iter = ArrayRef<Base>::const_iterator;
  llvm::iterator_range<const_base_iter> bases() const {
    return llvm::make_range(Bases, Bases + NumBases);
  }

  unsigned getNumBases() const { return NumBases; }
  const Base *getBase(unsigned I) const {
    assert(I < getNumBases());
    return &Bases[I];
  }

  using const_virtual_iter = ArrayRef<Base>::const_iterator;
  llvm::iterator_range<const_virtual_iter> virtual_bases() const {
    return llvm::make_range(VBases, VBases + NumVBases);
  }

  unsigned getNumVirtualBases() const { return NumVBases; }
  const Base *getVirtualBase(unsigned I) const { return &VBases[I]; }

  void dump(llvm::raw_ostream &OS, unsigned Indentation = 0,
            unsigned Offset = 0) const;
  void dump() const { dump(llvm::errs()); }

private:
  /// Constructor used by Program to create record descriptors.
  Record(const RecordDecl *, const Base *Bases, unsigned NumBases,
         const Field *Fields, unsigned NumFields, Base *VBases,
         unsigned NumVBases, unsigned VirtualSize, unsigned BaseSize);

private:
  friend class Program;

  /// Original declaration.
  const RecordDecl *Decl;
  /// List of all base classes.
  const Base *Bases;
  unsigned NumBases;
  /// List of all the fields in the record.
  const Field *Fields;
  unsigned NumFields;
  /// List of all virtual bases.
  Base *VBases;
  unsigned NumVBases;

  /// Mapping from declarations to bases.
  llvm::DenseMap<const RecordDecl *, const Base *> BaseMap;
  /// Mapping from field identifiers to descriptors.
  llvm::DenseMap<const FieldDecl *, const Field *> FieldMap;
  /// Mapping from declarations to virtual bases.
  llvm::DenseMap<const RecordDecl *, const Base *> VirtualBaseMap;
  /// Size of the structure.
  unsigned BaseSize;
  /// Size of all virtual bases.
  unsigned VirtualSize;
  /// If this record is a union.
  bool IsUnion;
  /// If this is an anonymous union.
  bool IsAnonymousUnion;
};

} // namespace interp
} // namespace clang

#endif
