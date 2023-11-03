//===- CIRTypes.h - MLIR CIR Types ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CIR_IR_CIRTYPES_H_
#define MLIR_DIALECT_CIR_IR_CIRTYPES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"

//===----------------------------------------------------------------------===//
// CIR Dialect Tablegen'd Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// CIR StructType
//
// The base type for all RecordDecls.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cir {

namespace detail {
struct StructTypeStorage;
} // namespace detail

/// Each unique clang::RecordDecl is mapped to a `cir.struct` and any object in
/// C/C++ that has a struct type will have a `cir.struct` in CIR.
class StructType
    : public Type::TypeBase<StructType, Type, detail::StructTypeStorage,
                            DataLayoutTypeInterface::Trait> {
  // FIXME(cir): migrate this type to Tablegen once mutable types are supported.
public:
  using Base::Base;
  using Base::getChecked;
  using Base::verify;

  static constexpr StringLiteral name = "cir.struct";

  enum RecordKind : uint32_t { Class, Union, Struct };

  /// Create a identified and complete struct type.
  static StructType get(MLIRContext *context, ArrayRef<Type> members,
                        StringAttr name, bool packed, RecordKind kind,
                        ASTRecordDeclInterface ast = {});
  static StructType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, ArrayRef<Type> members,
                               StringAttr name, bool packed, RecordKind kind,
                               ASTRecordDeclInterface ast = {});

  /// Create a identified and incomplete struct type.
  static StructType get(MLIRContext *context, StringAttr name, RecordKind kind);
  static StructType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, StringAttr name,
                               RecordKind kind);

  /// Create a anonymous struct type (always complete).
  static StructType get(MLIRContext *context, ArrayRef<Type> members,
                        bool packed, RecordKind kind,
                        ASTRecordDeclInterface ast = {});
  static StructType getChecked(function_ref<InFlightDiagnostic()> emitError,
                               MLIRContext *context, ArrayRef<Type> members,
                               bool packed, RecordKind kind,
                               ASTRecordDeclInterface ast = {});

  /// Validate the struct about to be constructed.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<Type> members, StringAttr name,
                              bool incomplete, bool packed,
                              StructType::RecordKind kind,
                              ASTRecordDeclInterface ast);

  // Parse/print methods.
  static constexpr StringLiteral getMnemonic() { return {"struct"}; }
  static Type parse(AsmParser &odsParser);
  void print(AsmPrinter &odsPrinter) const;

  // Accessors
  ASTRecordDeclInterface getAst() const;
  ArrayRef<Type> getMembers() const;
  StringAttr getName() const;
  StructType::RecordKind getKind() const;
  bool getIncomplete() const;
  bool getPacked() const;
  void dropAst();

  // Predicates
  bool isClass() const { return getKind() == RecordKind::Class; };
  bool isStruct() const { return getKind() == RecordKind::Struct; };
  bool isUnion() const { return getKind() == RecordKind::Union; };
  bool isComplete() const { return !isIncomplete(); };
  bool isIncomplete() const;

  // Utilities
  Type getLargestMember(const DataLayout &dataLayout) const;
  size_t getNumElements() const { return getMembers().size(); };
  std::string getKindAsStr() {
    switch (getKind()) {
    case RecordKind::Class:
      return "class";
    case RecordKind::Union:
      return "union";
    case RecordKind::Struct:
      return "struct";
    }
  }
  std::string getPrefixedName() {
    return getKindAsStr() + "." + getName().getValue().str();
  }

  /// DataLayoutTypeInterface methods.
  llvm::TypeSize getTypeSizeInBits(const DataLayout &dataLayout,
                                   DataLayoutEntryListRef params) const;
  uint64_t getABIAlignment(const DataLayout &dataLayout,
                           DataLayoutEntryListRef params) const;
  uint64_t getPreferredAlignment(const DataLayout &dataLayout,
                                 DataLayoutEntryListRef params) const;

  // Utilities for lazily computing and cacheing data layout info.
private:
  mutable Type largestMember{};
  mutable std::optional<bool> padded{};
  mutable std::optional<unsigned> size{}, align{};
  bool isPadded(const DataLayout &dataLayout) const;
  void computeSizeAndAlignment(const DataLayout &dataLayout) const;
};

} // namespace cir
} // namespace mlir

#endif // MLIR_DIALECT_CIR_IR_CIRTYPES_H_
