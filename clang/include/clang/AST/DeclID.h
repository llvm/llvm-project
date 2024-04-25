//===--- DeclID.h - ID number for deserialized declarations  ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines DeclID class family to describe the deserialized
// declarations. The DeclID is widely used in AST via LazyDeclPtr, or calls to
// `ExternalASTSource::getExternalDecl`. It will be helpful for type safety to
// require the use of `DeclID` to explicit.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLID_H
#define LLVM_CLANG_AST_DECLID_H

#include "llvm/ADT/iterator.h"

namespace clang {

/// Predefined declaration IDs.
///
/// These declaration IDs correspond to predefined declarations in the AST
/// context, such as the NULL declaration ID. Such declarations are never
/// actually serialized, since they will be built by the AST context when
/// it is created.
enum PredefinedDeclIDs {
  /// The NULL declaration.
  PREDEF_DECL_NULL_ID = 0,

  /// The translation unit.
  PREDEF_DECL_TRANSLATION_UNIT_ID = 1,

  /// The Objective-C 'id' type.
  PREDEF_DECL_OBJC_ID_ID = 2,

  /// The Objective-C 'SEL' type.
  PREDEF_DECL_OBJC_SEL_ID = 3,

  /// The Objective-C 'Class' type.
  PREDEF_DECL_OBJC_CLASS_ID = 4,

  /// The Objective-C 'Protocol' type.
  PREDEF_DECL_OBJC_PROTOCOL_ID = 5,

  /// The signed 128-bit integer type.
  PREDEF_DECL_INT_128_ID = 6,

  /// The unsigned 128-bit integer type.
  PREDEF_DECL_UNSIGNED_INT_128_ID = 7,

  /// The internal 'instancetype' typedef.
  PREDEF_DECL_OBJC_INSTANCETYPE_ID = 8,

  /// The internal '__builtin_va_list' typedef.
  PREDEF_DECL_BUILTIN_VA_LIST_ID = 9,

  /// The internal '__va_list_tag' struct, if any.
  PREDEF_DECL_VA_LIST_TAG = 10,

  /// The internal '__builtin_ms_va_list' typedef.
  PREDEF_DECL_BUILTIN_MS_VA_LIST_ID = 11,

  /// The predeclared '_GUID' struct.
  PREDEF_DECL_BUILTIN_MS_GUID_ID = 12,

  /// The extern "C" context.
  PREDEF_DECL_EXTERN_C_CONTEXT_ID = 13,

  /// The internal '__make_integer_seq' template.
  PREDEF_DECL_MAKE_INTEGER_SEQ_ID = 14,

  /// The internal '__NSConstantString' typedef.
  PREDEF_DECL_CF_CONSTANT_STRING_ID = 15,

  /// The internal '__NSConstantString' tag type.
  PREDEF_DECL_CF_CONSTANT_STRING_TAG_ID = 16,

  /// The internal '__type_pack_element' template.
  PREDEF_DECL_TYPE_PACK_ELEMENT_ID = 17,
};

/// The number of declaration IDs that are predefined.
///
/// For more information about predefined declarations, see the
/// \c PredefinedDeclIDs type and the PREDEF_DECL_*_ID constants.
const unsigned int NUM_PREDEF_DECL_IDS = 18;

/// An ID number that refers to a declaration in an AST file.
///
/// The ID numbers of declarations are consecutive (in order of
/// discovery), with values below NUM_PREDEF_DECL_IDS being reserved.
/// At the start of a chain of precompiled headers, declaration ID 1 is
/// used for the translation unit declaration.
using DeclID = uint32_t;

class LocalDeclID {
public:
  explicit LocalDeclID(DeclID ID) : ID(ID) {}

  DeclID get() const { return ID; }

private:
  DeclID ID;
};

/// Wrapper class for DeclID. This is helpful to not mix the use of LocalDeclID
/// and GlobalDeclID to improve the type safety.
class GlobalDeclID {
public:
  GlobalDeclID() : ID(PREDEF_DECL_NULL_ID) {}
  explicit GlobalDeclID(DeclID ID) : ID(ID) {}

  DeclID get() const { return ID; }

  explicit operator DeclID() const { return ID; }

  friend bool operator==(const GlobalDeclID &LHS, const GlobalDeclID &RHS) {
    return LHS.ID == RHS.ID;
  }
  friend bool operator!=(const GlobalDeclID &LHS, const GlobalDeclID &RHS) {
    return LHS.ID != RHS.ID;
  }
  // We may sort the global decl ID.
  friend bool operator<(const GlobalDeclID &LHS, const GlobalDeclID &RHS) {
    return LHS.ID < RHS.ID;
  }
  friend bool operator>(const GlobalDeclID &LHS, const GlobalDeclID &RHS) {
    return LHS.ID > RHS.ID;
  }
  friend bool operator<=(const GlobalDeclID &LHS, const GlobalDeclID &RHS) {
    return LHS.ID <= RHS.ID;
  }
  friend bool operator>=(const GlobalDeclID &LHS, const GlobalDeclID &RHS) {
    return LHS.ID >= RHS.ID;
  }

private:
  DeclID ID;
};

/// A helper iterator adaptor to convert the iterators to `SmallVector<DeclID>`
/// to the iterators to `SmallVector<GlobalDeclID>`.
class GlobalDeclIDIterator
    : public llvm::iterator_adaptor_base<GlobalDeclIDIterator, const DeclID *,
                                         std::forward_iterator_tag,
                                         GlobalDeclID> {
public:
  GlobalDeclIDIterator() : iterator_adaptor_base(nullptr) {}

  GlobalDeclIDIterator(const DeclID *ID) : iterator_adaptor_base(ID) {}

  value_type operator*() const { return GlobalDeclID(*I); }

  bool operator==(const GlobalDeclIDIterator &RHS) const { return I == RHS.I; }
};

/// A helper iterator adaptor to convert the iterators to
/// `SmallVector<GlobalDeclID>` to the iterators to `SmallVector<DeclID>`.
class DeclIDIterator
    : public llvm::iterator_adaptor_base<DeclIDIterator, const GlobalDeclID *,
                                         std::forward_iterator_tag, DeclID> {
public:
  DeclIDIterator() : iterator_adaptor_base(nullptr) {}

  DeclIDIterator(const GlobalDeclID *ID) : iterator_adaptor_base(ID) {}

  value_type operator*() const { return DeclID(*I); }

  bool operator==(const DeclIDIterator &RHS) const { return I == RHS.I; }
};

} // namespace clang

#endif
