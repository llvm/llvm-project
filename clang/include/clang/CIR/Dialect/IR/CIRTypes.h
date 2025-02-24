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
#include "clang/CIR/Interfaces/CIRFPTypeInterface.h"

#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"

#include "clang/CIR/Dialect/IR/CIROpsEnums.h"

//===----------------------------------------------------------------------===//
// CIR StructType
//
// The base type for all RecordDecls.
//===----------------------------------------------------------------------===//

namespace cir {

namespace detail {
struct StructTypeStorage;
} // namespace detail

/// Each unique clang::RecordDecl is mapped to a `cir.struct` and any object in
/// C/C++ that has a struct type will have a `cir.struct` in CIR.
///
/// There are three possible formats for this type:
///
///  - Identified and complete structs: unique name and a known body.
///  - Identified and incomplete structs: unique name and unknown body.
///  - Anonymous structs: no name and a known body.
///
/// Identified structs are uniqued by their name, and anonymous structs are
/// uniqued by their body. This means that two anonymous structs with the same
/// body will be the same type, and two identified structs with the same name
/// will be the same type. Attempting to build a struct with an existing name,
/// but a different body will result in an error.
///
/// A few examples:
///
/// ```mlir
///     !complete = !cir.struct<struct "complete" {!cir.int<u, 8>}>
///     !incomplete = !cir.struct<struct "incomplete" incomplete>
///     !anonymous = !cir.struct<struct {!cir.int<u, 8>}>
/// ```
///
/// Incomplete structs are mutable, meaning they can be later completed with a
/// body automatically updating in place every type in the code that uses the
/// incomplete struct. Mutability allows for recursive types to be represented,
/// meaning the struct can have members that refer to itself. This is useful for
/// representing recursive records and is implemented through a special syntax.
/// In the example below, the `Node` struct has a member that is a pointer to a
/// `Node` struct:
///
/// ```mlir
///     !struct = !cir.struct<struct "Node" {!cir.ptr<!cir.struct<struct
///     "Node">>}>
/// ```
class StructType
    : public mlir::Type::TypeBase<
          StructType, mlir::Type, detail::StructTypeStorage,
          mlir::DataLayoutTypeInterface::Trait, mlir::TypeTrait::IsMutable> {
  // FIXME(cir): migrate this type to Tablegen once mutable types are supported.
public:
  using Base::Base;
  using Base::getChecked;
  using Base::verifyInvariants;

  static constexpr llvm::StringLiteral name = "cir.struct";

  enum RecordKind : uint32_t { Class, Union, Struct };

  /// Create an identified and complete struct type.
  static StructType get(mlir::MLIRContext *context,
                        llvm::ArrayRef<mlir::Type> members,
                        mlir::StringAttr name, bool packed, bool padded,
                        RecordKind kind, ASTRecordDeclInterface ast = {});
  static StructType
  getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
             mlir::MLIRContext *context, llvm::ArrayRef<mlir::Type> members,
             mlir::StringAttr name, bool packed, bool padded, RecordKind kind,
             ASTRecordDeclInterface ast = {});

  /// Create an identified and incomplete struct type.
  static StructType get(mlir::MLIRContext *context, mlir::StringAttr name,
                        RecordKind kind);
  static StructType
  getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
             mlir::MLIRContext *context, mlir::StringAttr name,
             RecordKind kind);

  /// Create an anonymous struct type (always complete).
  static StructType get(mlir::MLIRContext *context,
                        llvm::ArrayRef<mlir::Type> members, bool packed,
                        bool padded, RecordKind kind,
                        ASTRecordDeclInterface ast = {});
  static StructType
  getChecked(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
             mlir::MLIRContext *context, llvm::ArrayRef<mlir::Type> members,
             bool packed, bool padded, RecordKind kind,
             ASTRecordDeclInterface ast = {});

  /// Validate the struct about to be constructed.
  static llvm::LogicalResult
  verifyInvariants(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   llvm::ArrayRef<mlir::Type> members, mlir::StringAttr name,
                   bool incomplete, bool packed, bool padded,
                   StructType::RecordKind kind, ASTRecordDeclInterface ast);

  // Parse/print methods.
  static constexpr llvm::StringLiteral getMnemonic() { return {"struct"}; }
  static mlir::Type parse(mlir::AsmParser &odsParser);
  void print(mlir::AsmPrinter &odsPrinter) const;

  // Accessors
  ASTRecordDeclInterface getAst() const;
  llvm::ArrayRef<mlir::Type> getMembers() const;
  mlir::StringAttr getName() const;
  StructType::RecordKind getKind() const;
  bool getIncomplete() const;
  bool getPacked() const;
  bool getPadded() const;
  void dropAst();

  // Predicates
  bool isClass() const { return getKind() == RecordKind::Class; };
  bool isStruct() const { return getKind() == RecordKind::Struct; };
  bool isUnion() const { return getKind() == RecordKind::Union; };
  bool isComplete() const { return !isIncomplete(); };
  bool isIncomplete() const;

  // Utilities
  mlir::Type getLargestMember(const mlir::DataLayout &dataLayout) const;
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
    llvm_unreachable("Invalid value for StructType::getKind()");
  }
  std::string getPrefixedName() {
    return getKindAsStr() + "." + getName().getValue().str();
  }

  /// Complete the struct type by mutating its members and attributes.
  void complete(llvm::ArrayRef<mlir::Type> members, bool packed, bool isPadded,
                ASTRecordDeclInterface ast = {});

  /// DataLayoutTypeInterface methods.
  llvm::TypeSize getTypeSizeInBits(const mlir::DataLayout &dataLayout,
                                   mlir::DataLayoutEntryListRef params) const;
  uint64_t getABIAlignment(const mlir::DataLayout &dataLayout,
                           mlir::DataLayoutEntryListRef params) const;
  uint64_t getPreferredAlignment(const mlir::DataLayout &dataLayout,
                                 mlir::DataLayoutEntryListRef params) const;
  uint64_t getElementOffset(const mlir::DataLayout &dataLayout,
                            unsigned idx) const;

  bool isLayoutIdentical(const StructType &other);

  // Utilities for lazily computing and cacheing data layout info.
private:
  // FIXME: currently opaque because there's a cycle if CIRTypes.types include
  // from CIRAttrs.h. The implementation operates in terms of StructLayoutAttr
  // instead.
  mutable mlir::Attribute layoutInfo;
  void computeSizeAndAlignment(const mlir::DataLayout &dataLayout) const;
};

bool isAnyFloatingPointType(mlir::Type t);
bool isFPOrFPVectorTy(mlir::Type);
bool isIntOrIntVectorTy(mlir::Type);
} // namespace cir

mlir::ParseResult parseAddrSpaceAttribute(mlir::AsmParser &p,
                                          mlir::Attribute &addrSpaceAttr);
void printAddrSpaceAttribute(mlir::AsmPrinter &p,
                             mlir::Attribute addrSpaceAttr);

//===----------------------------------------------------------------------===//
// CIR Dialect Tablegen'd Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsTypes.h.inc"

#endif // MLIR_DIALECT_CIR_IR_CIRTYPES_H_
