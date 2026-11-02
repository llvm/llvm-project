//===-- FIROpPatterns.h -- FIR operation conversion patterns ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_CODEGEN_FIROPPATTERNS_H
#define FORTRAN_OPTIMIZER_CODEGEN_FIROPPATTERNS_H

#include "flang/Optimizer/CodeGen/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include <type_traits>

namespace fir {

struct FIRToLLVMPassOptions;

static constexpr unsigned defaultAddressSpace = 0u;

/// Symbol lookups in the FIR to LLVM conversion.
///
/// The conversion patterns resolve `fir.address_of`, derived type descriptor
/// globals, comdat selectors and runtime declarations by name.
/// `mlir::SymbolTable::lookupSymbolIn` walks every operation of the symbol
/// table operation, so resolving each reference that way costs O(#symbols),
/// and converting a module with N derived types (about 25 runtime type info
/// globals each) takes O(N^2). When the driver provides a
/// `mlir::SymbolTableCollection`, these helpers resolve names through it in
/// O(1) and keep it current while the conversion replaces symbol operations;
/// without one they fall back to the linear walk.
///
/// A driver that provides a collection must:
///  - build the tables with `buildSymbolTables` before applying the conversion.
///    With pattern rollback enabled (the default), a replaced `fir.global` is
///    erased only when the conversion finishes, so a table constructed lazily
///    in the middle of the conversion would see both the `fir.global` and the
///    `llvm.mlir.global` of the same name and silently keep the first one;
///  - pass the same collection to every pattern set of the conversion that
///    looks up or declares symbols the FIR patterns also use, in particular
///    `mlir::populateFuncToLLVMConversionPatterns` (so that `func.func`
///    replacements are reflected: the allocator helpers and
///    `getDependentTypeMemSizeFn` look functions up by name) and, where used,
///    `mlir::populateFinalizeMemRefToLLVMConversionPatterns` (which declares
///    malloc/free itself). A declaration made behind the collection's back is
///    invisible to the FIR patterns, which would then add a second one.
///
/// Symbol operations replaced by other pattern sets of the same conversion
/// (OpenMP region ops, for instance) stay in the collection as stale entries.
/// Nothing in the FIR patterns looks those up, and the collection does not
/// outlive the conversion.
///
/// Registering an operation created through the rewriter is only valid if the
/// pattern cannot be rolled back afterwards: the pattern must not fail after
/// registering, and every operation it creates or modifies in place must be
/// legalizable, since a failed legalization of those rolls the pattern back
/// and erases what it created, leaving a dangling entry. This holds for the
/// sites below that register rewriter-created symbols (GlobalOpConversion,
/// genSourceFile, the external type descriptor declaration in
/// getTypeDescriptor): they create LLVM dialect operations only, after their
/// last failure path.

/// Look up the symbol `name` in `symbolTableOp`, returning null if there is no
/// such symbol or it is not an `OpTy`. `name` is a string (`StringRef` or
/// anything convertible to it), a `StringAttr`, or a `SymbolRefAttr`; a nested
/// reference is resolved through the nested symbol tables.
template <typename OpTy, typename NameT>
OpTy lookupSymbolIn(mlir::Operation *symbolTableOp, NameT name,
                    mlir::SymbolTableCollection *symbolTables) {
  if (!symbolTables)
    return mlir::dyn_cast_or_null<OpTy>(
        mlir::SymbolTable::lookupSymbolIn(symbolTableOp, name));
  OpTy op;
  if constexpr (std::is_same_v<NameT, mlir::SymbolRefAttr>)
    op = symbolTables->lookupSymbolIn<OpTy>(symbolTableOp, name);
  else
    op = symbolTables->getSymbolTable(symbolTableOp).lookup<OpTy>(name);
#ifdef EXPENSIVE_CHECKS
  // A replacement is always inserted before the operation it replaces, so on
  // a hit the linear walk must find the same operation.
  assert((!op ||
          op == mlir::dyn_cast_or_null<OpTy>(
                    mlir::SymbolTable::lookupSymbolIn(symbolTableOp, name))) &&
         "symbol tables and module out of sync");
#endif
  return op;
}

/// Record `symbol`, a symbol operation just created inside `symbolTableOp`, in
/// the symbol tables. No-op without a collection.
void registerSymbol(mlir::SymbolTableCollection *symbolTables,
                    mlir::Operation *symbolTableOp, mlir::Operation *symbol);

/// Record that `newSymbol` replaces `oldSymbol` (both children of
/// `symbolTableOp`, with the same name). Must be called before the rewriter
/// erases `oldSymbol`. No-op without a collection.
void replaceSymbol(mlir::SymbolTableCollection *symbolTables,
                   mlir::Operation *symbolTableOp, mlir::Operation *oldSymbol,
                   mlir::Operation *newSymbol);

/// Forget `symbol`, which is about to be erased. No-op without a collection.
void unregisterSymbol(mlir::SymbolTableCollection *symbolTables,
                      mlir::Operation *symbolTableOp, mlir::Operation *symbol);

/// Build the symbol tables of `root` and of every symbol table operation
/// nested in it (gpu modules, comdats). See the note above on why this must
/// run before the conversion. A pass instance scheduled on a nested symbol
/// table operation (a `gpu.module`) must pass that operation, not the parent
/// module: sibling operations may be converted concurrently.
void buildSymbolTables(mlir::SymbolTableCollection &symbolTables,
                       mlir::Operation *root);

class ConvertFIRToLLVMPattern : public mlir::ConvertToLLVMPattern {
public:
  ConvertFIRToLLVMPattern(llvm::StringRef rootOpName,
                          mlir::MLIRContext *context,
                          const fir::LLVMTypeConverter &typeConverter,
                          const fir::FIRToLLVMPassOptions &options,
                          mlir::SymbolTableCollection *symbolTables = nullptr,
                          mlir::PatternBenefit benefit = 1);

protected:
  mlir::Type convertType(mlir::Type ty) const {
    return lowerTy().convertType(ty);
  }

  // Convert FIR type to LLVM without turning fir.box<T> into memory
  // reference.
  mlir::Type convertObjectType(mlir::Type firType) const;

  mlir::LLVM::ConstantOp
  genI32Constant(mlir::Location loc, mlir::ConversionPatternRewriter &rewriter,
                 int value) const;

  mlir::LLVM::ConstantOp
  genConstantOffset(mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter,
                    int offset) const;

  /// Perform an extension or truncation as needed on an integer value. Lowering
  /// to the specific target may involve some sign-extending or truncation of
  /// values, particularly to fit them from abstract box types to the
  /// appropriate reified structures.
  mlir::Value integerCast(mlir::Location loc,
                          mlir::ConversionPatternRewriter &rewriter,
                          mlir::Type ty, mlir::Value val,
                          bool fold = false) const;

  struct TypePair {
    mlir::Type fir;
    mlir::Type llvm;
  };

  TypePair getBoxTypePair(mlir::Type firBoxTy) const;

  /// Construct code sequence to extract the specific value from a `fir.box`.
  mlir::Value getValueFromBox(mlir::Location loc, TypePair boxTy,
                              mlir::Value box, mlir::Type resultTy,
                              mlir::ConversionPatternRewriter &rewriter,
                              int boxValue) const;

  /// Method to construct code sequence to get the triple for dimension `dim`
  /// from a box.
  llvm::SmallVector<mlir::Value, 3>
  getDimsFromBox(mlir::Location loc, llvm::ArrayRef<mlir::Type> retTys,
                 TypePair boxTy, mlir::Value box, mlir::Value dim,
                 mlir::ConversionPatternRewriter &rewriter) const;

  llvm::SmallVector<mlir::Value, 3>
  getDimsFromBox(mlir::Location loc, llvm::ArrayRef<mlir::Type> retTys,
                 TypePair boxTy, mlir::Value box, int dim,
                 mlir::ConversionPatternRewriter &rewriter) const;

  mlir::Value
  loadDimFieldFromBox(mlir::Location loc, TypePair boxTy, mlir::Value box,
                      mlir::Value dim, int off, mlir::Type ty,
                      mlir::ConversionPatternRewriter &rewriter) const;

  mlir::Value
  getDimFieldFromBox(mlir::Location loc, TypePair boxTy, mlir::Value box,
                     int dim, int off, mlir::Type ty,
                     mlir::ConversionPatternRewriter &rewriter) const;

  mlir::Value getStrideFromBox(mlir::Location loc, TypePair boxTy,
                               mlir::Value box, unsigned dim,
                               mlir::ConversionPatternRewriter &rewriter) const;

  /// Read base address from a fir.box. Returned address has type ty.
  mlir::Value
  getBaseAddrFromBox(mlir::Location loc, TypePair boxTy, mlir::Value box,
                     mlir::ConversionPatternRewriter &rewriter) const;

  mlir::Value
  getElementSizeFromBox(mlir::Location loc, mlir::Type resultTy, TypePair boxTy,
                        mlir::Value box,
                        mlir::ConversionPatternRewriter &rewriter) const;

  mlir::Value getRankFromBox(mlir::Location loc, TypePair boxTy,
                             mlir::Value box,
                             mlir::ConversionPatternRewriter &rewriter) const;

  mlir::Value getExtraFromBox(mlir::Location loc, TypePair boxTy,
                              mlir::Value box,
                              mlir::ConversionPatternRewriter &rewriter) const;

  // Get the element type given an LLVM type that is of the form
  // (array|struct|vector)+ and the provided indexes.
  mlir::Type getBoxEleTy(mlir::Type type,
                         llvm::ArrayRef<std::int64_t> indexes) const;

  // Return LLVM type of the object described by a fir.box of \p boxType.
  mlir::Type getLlvmObjectTypeFromBoxType(mlir::Type boxType) const;

  /// Read the address of the type descriptor from a box.
  mlir::Value
  loadTypeDescAddress(mlir::Location loc, TypePair boxTy, mlir::Value box,
                      mlir::ConversionPatternRewriter &rewriter) const;

  // Load the attribute from the \p box and perform a check against \p maskValue
  // The final comparison is implemented as `(attribute & maskValue) != 0`.
  mlir::Value genBoxAttributeCheck(mlir::Location loc, TypePair boxTy,
                                   mlir::Value box,
                                   mlir::ConversionPatternRewriter &rewriter,
                                   unsigned maskValue) const;

  /// Compute the descriptor size in bytes. The result is not guaranteed to be a
  /// compile time constant if the box is for an assumed rank, in which case the
  /// box rank will be read.
  mlir::Value computeBoxSize(mlir::Location, TypePair boxTy, mlir::Value box,
                             mlir::ConversionPatternRewriter &rewriter) const;

  template <typename... ARGS>
  mlir::LLVM::GEPOp genGEP(mlir::Location loc, mlir::Type ty,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Value base, ARGS... args) const {
    llvm::SmallVector<mlir::LLVM::GEPArg> cv = {args...};
    auto llvmPtrTy =
        mlir::LLVM::LLVMPointerType::get(ty.getContext(), /*addressSpace=*/0);
    return mlir::LLVM::GEPOp::create(rewriter, loc, llvmPtrTy, ty, base, cv);
  }

  // Find the Block in which the alloca should be inserted.
  // The order to recursively find the proper block:
  // 1. An OpenMP Op that will be outlined.
  // 2. An OpenMP or OpenACC Op with one or more regions holding executable
  // code.
  // 3. A LLVMFuncOp
  // 4. The first ancestor that is one of the above.
  mlir::Block *getBlockForAllocaInsert(mlir::Operation *op,
                                       mlir::Region *parentRegion) const;

  // Generate an alloca of size 1 for an object of type \p llvmObjectTy in the
  // allocation address space provided for the architecture in the DataLayout
  // specification. If the address space is different from the devices
  // program address space we perform a cast. In the case of most architectures
  // the program and allocation address space will be the default of 0 and no
  // cast will be emitted.
  mlir::Value
  genAllocaAndAddrCastWithType(mlir::Location loc, mlir::Type llvmObjectTy,
                               unsigned alignment,
                               mlir::ConversionPatternRewriter &rewriter) const;

  const fir::LLVMTypeConverter &lowerTy() const {
    return *static_cast<const fir::LLVMTypeConverter *>(
        this->getTypeConverter());
  }

  const mlir::DataLayout &getDataLayout() const {
    return lowerTy().getDataLayout();
  }

  void attachTBAATag(mlir::LLVM::AliasAnalysisOpInterface op,
                     mlir::Type baseFIRType, mlir::Type accessFIRType,
                     mlir::LLVM::GEPOp gep) const {
    lowerTy().attachTBAATag(op, baseFIRType, accessFIRType, gep);
  }

  unsigned
  getAllocaAddressSpace(mlir::ConversionPatternRewriter &rewriter) const;

  unsigned
  getProgramAddressSpace(mlir::ConversionPatternRewriter &rewriter) const;

  unsigned
  getGlobalAddressSpace(mlir::ConversionPatternRewriter &rewriter) const;

  const fir::FIRToLLVMPassOptions &options;

  /// Symbol tables of the module being converted, maintained by the patterns
  /// as they replace symbol operations (see `fir::lookupSymbolIn`). Null when
  /// the driver did not provide any; lookups then fall back to a linear walk.
  mlir::SymbolTableCollection *symbolTables;

  using ConvertToLLVMPattern::matchAndRewrite;
};

template <typename SourceOp>
class FIROpConversion : public ConvertFIRToLLVMPattern {
public:
  using OpAdaptor = typename SourceOp::Adaptor;
  using OneToNOpAdaptor = typename SourceOp::template GenericAdaptor<
      mlir::ArrayRef<mlir::ValueRange>>;

  explicit FIROpConversion(const LLVMTypeConverter &typeConverter,
                           const fir::FIRToLLVMPassOptions &options,
                           mlir::SymbolTableCollection *symbolTables = nullptr,
                           mlir::PatternBenefit benefit = 1)
      : ConvertFIRToLLVMPattern(SourceOp::getOperationName(),
                                &typeConverter.getContext(), typeConverter,
                                options, symbolTables, benefit) {}

  /// Wrappers around the RewritePattern methods that pass the derived op type.
  llvm::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    return matchAndRewrite(mlir::cast<SourceOp>(op),
                           OpAdaptor(operands, mlir::cast<SourceOp>(op)),
                           rewriter);
  }
  llvm::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::ArrayRef<mlir::ValueRange> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto sourceOp = mlir::cast<SourceOp>(op);
    return matchAndRewrite(sourceOp, OneToNOpAdaptor(operands, sourceOp),
                           rewriter);
  }
  /// Methods that operate on the SourceOp type. These must be
  /// overridden by the derived pattern class.
  virtual llvm::LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("matchAndRewrite is not implemented");
  }
  virtual llvm::LogicalResult
  matchAndRewrite(SourceOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const {
    return dispatchTo1To1(*this, op, adaptor, rewriter);
  }

private:
  using ConvertFIRToLLVMPattern::matchAndRewrite;
};

/// FIR conversion pattern template
template <typename FromOp>
class FIROpAndTypeConversion : public FIROpConversion<FromOp> {
public:
  using FIROpConversion<FromOp>::FIROpConversion;
  using OpAdaptor = typename FromOp::Adaptor;

  llvm::LogicalResult
  matchAndRewrite(FromOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Type ty = this->convertType(op.getType());
    return doRewrite(op, ty, adaptor, rewriter);
  }

  virtual llvm::LogicalResult
  doRewrite(FromOp addr, mlir::Type ty, OpAdaptor adaptor,
            mlir::ConversionPatternRewriter &rewriter) const = 0;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_FIROPPATTERNS_H
