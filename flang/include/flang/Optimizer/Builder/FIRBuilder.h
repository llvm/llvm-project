//===-- FirBuilder.h -- FIR operation builder -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_FIRBUILDER_H
#define FORTRAN_OPTIMIZER_BUILDER_FIRBUILDER_H

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Support/MathOptionsBase.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>
#include <utility>

namespace mlir {
class DataLayout;
class SymbolTable;
}

namespace fir {
class AbstractArrayBox;
class ExtendedValue;
class MutableBoxValue;
class BoxValue;

/// Get the integer type with a pointer size.
inline mlir::Type getIntPtrType(mlir::OpBuilder &builder) {
  // TODO: Delay the need of such type until codegen or find a way to use
  // llvm::DataLayout::getPointerSizeInBits here.
  return builder.getI64Type();
}

//===----------------------------------------------------------------------===//
// FirOpBuilder
//===----------------------------------------------------------------------===//

/// Extends the MLIR OpBuilder to provide methods for building common FIR
/// patterns.
class FirOpBuilder : public mlir::OpBuilder, public mlir::OpBuilder::Listener {
public:
  explicit FirOpBuilder(mlir::Operation *op, fir::KindMapping kindMap,
                        mlir::SymbolTable *symbolTable = nullptr)
      : OpBuilder{op, /*listener=*/this}, kindMap{std::move(kindMap)},
        symbolTable{symbolTable} {
    auto fmi = mlir::dyn_cast<mlir::arith::ArithFastMathInterface>(*op);
    if (fmi) {
      // Set the builder with FastMathFlags attached to the operation.
      setFastMathFlags(fmi.getFastMathFlagsAttr().getValue());
    }
  }
  explicit FirOpBuilder(mlir::OpBuilder &builder, fir::KindMapping kindMap,
                        mlir::SymbolTable *symbolTable = nullptr)
      : OpBuilder(builder), OpBuilder::Listener(), kindMap{std::move(kindMap)},
        symbolTable{symbolTable} {
    setListener(this);
  }
  explicit FirOpBuilder(mlir::OpBuilder &builder, mlir::ModuleOp mod)
      : OpBuilder(builder), OpBuilder::Listener(),
        kindMap{getKindMapping(mod)} {
    setListener(this);
  }
  explicit FirOpBuilder(mlir::OpBuilder &builder, fir::KindMapping kindMap,
                        mlir::Operation *op)
      : OpBuilder(builder), OpBuilder::Listener(), kindMap{std::move(kindMap)} {
    setListener(this);
    auto fmi = mlir::dyn_cast<mlir::arith::ArithFastMathInterface>(*op);
    if (fmi) {
      // Set the builder with FastMathFlags attached to the operation.
      setFastMathFlags(fmi.getFastMathFlagsAttr().getValue());
    }
  }
  FirOpBuilder(mlir::OpBuilder &builder, mlir::Operation *op)
      : FirOpBuilder(builder, fir::getKindMapping(op), op) {}

  // The listener self-reference has to be updated in case of copy-construction.
  FirOpBuilder(const FirOpBuilder &other)
      : OpBuilder(other), OpBuilder::Listener(), kindMap{other.kindMap},
        fastMathFlags{other.fastMathFlags},
        integerOverflowFlags{other.integerOverflowFlags},
        symbolTable{other.symbolTable} {
    setListener(this);
  }

  FirOpBuilder(FirOpBuilder &&other)
      : OpBuilder(other), OpBuilder::Listener(),
        kindMap{std::move(other.kindMap)}, fastMathFlags{other.fastMathFlags},
        integerOverflowFlags{other.integerOverflowFlags},
        symbolTable{other.symbolTable} {
    setListener(this);
  }

  /// Get the current Region of the insertion point.
  mlir::Region &getRegion() { return *getBlock()->getParent(); }

  /// Get the current Module
  mlir::ModuleOp getModule() {
    return getRegion().getParentOfType<mlir::ModuleOp>();
  }

  /// Get the current Function
  mlir::func::FuncOp getFunction() {
    return getRegion().getParentOfType<mlir::func::FuncOp>();
  }

  /// Get a reference to the kind map.
  const fir::KindMapping &getKindMap() { return kindMap; }

  /// Get func.func/fir.global symbol table attached to this builder if any.
  mlir::SymbolTable *getMLIRSymbolTable() { return symbolTable; }

  /// Get the default integer type
  [[maybe_unused]] mlir::IntegerType getDefaultIntegerType() {
    return getIntegerType(
        getKindMap().getIntegerBitsize(getKindMap().defaultIntegerKind()));
  }

  /// The LHS and RHS are not always in agreement in terms of type. In some
  /// cases, the disagreement is between COMPLEX and other scalar types. In that
  /// case, the conversion must insert (extract) out of a COMPLEX value to have
  /// the proper semantics and be strongly typed. E.g., converting an integer
  /// (real) to a complex, the real part is filled using the integer (real)
  /// after type conversion and the imaginary part is zero.
  mlir::Value convertWithSemantics(mlir::Location loc, mlir::Type toTy,
                                   mlir::Value val,
                                   bool allowCharacterConversion = false,
                                   bool allowRebox = false);

  /// Get the entry block of the current Function
  mlir::Block *getEntryBlock() { return &getFunction().front(); }

  /// Get the block for adding Allocas. If OpenMP is enabled then get the
  /// the alloca block from an Operation which can be Outlined. Otherwise
  /// use the entry block of the current Function
  mlir::Block *getAllocaBlock();

  /// Safely create a reference type to the type `eleTy`.
  mlir::Type getRefType(mlir::Type eleTy, bool isVolatile = false);

  /// Create a sequence of `eleTy` with `rank` dimensions of unknown size.
  mlir::Type getVarLenSeqTy(mlir::Type eleTy, unsigned rank = 1);

  /// Get character length type.
  mlir::Type getCharacterLengthType() { return getIndexType(); }

  /// Get the integer type whose bit width corresponds to the width of pointer
  /// types, or is bigger.
  mlir::Type getIntPtrType() { return fir::getIntPtrType(*this); }

  /// Wrap `str` to a SymbolRefAttr.
  mlir::SymbolRefAttr getSymbolRefAttr(llvm::StringRef str) {
    return mlir::SymbolRefAttr::get(getContext(), str);
  }

  /// Get the mlir float type that implements Fortran REAL(kind).
  mlir::Type getRealType(int kind);

  fir::BoxProcType getBoxProcType(mlir::FunctionType funcTy) {
    return fir::BoxProcType::get(getContext(), funcTy);
  }

  /// Create a null constant memory reference of type \p ptrType.
  /// If \p ptrType is not provided, !fir.ref<none> type will be used.
  mlir::Value createNullConstant(mlir::Location loc, mlir::Type ptrType = {});

  /// Create an integer constant of type \p type and value \p i.
  /// Should not be used with negative values with integer types of more
  /// than 64 bits.
  mlir::Value createIntegerConstant(mlir::Location loc, mlir::Type integerType,
                                    std::int64_t i);

  /// Create an integer of \p integerType where all the bits have been set to
  /// ones. Safe to use regardless of integerType bitwidth.
  mlir::Value createAllOnesInteger(mlir::Location loc, mlir::Type integerType);

  /// Create -1 constant of \p integerType. Safe to use regardless of
  /// integerType bitwidth.
  mlir::Value createMinusOneInteger(mlir::Location loc,
                                    mlir::Type integerType) {
    return createAllOnesInteger(loc, integerType);
  }

  /// Create a real constant from an integer value.
  mlir::Value createRealConstant(mlir::Location loc, mlir::Type realType,
                                 llvm::APFloat::integerPart val);

  /// Create a real constant from an APFloat value.
  mlir::Value createRealConstant(mlir::Location loc, mlir::Type realType,
                                 const llvm::APFloat &val);

  /// Create a real constant of type \p realType with a value zero.
  mlir::Value createRealZeroConstant(mlir::Location loc, mlir::Type realType) {
    return createRealConstant(loc, realType, 0u);
  }

  /// Create a slot for a local on the stack. Besides the variable's type and
  /// shape, it may be given name, pinned, or target attributes.
  mlir::Value allocateLocal(mlir::Location loc, mlir::Type ty,
                            llvm::StringRef uniqName, llvm::StringRef name,
                            bool pinned, llvm::ArrayRef<mlir::Value> shape,
                            llvm::ArrayRef<mlir::Value> lenParams,
                            bool asTarget = false);
  mlir::Value allocateLocal(mlir::Location loc, mlir::Type ty,
                            llvm::StringRef uniqName, llvm::StringRef name,
                            llvm::ArrayRef<mlir::Value> shape,
                            llvm::ArrayRef<mlir::Value> lenParams,
                            bool asTarget = false);

  /// Create a two dimensional ArrayAttr containing integer data as
  /// IntegerAttrs, effectively: ArrayAttr<ArrayAttr<IntegerAttr>>>.
  mlir::ArrayAttr create2DI64ArrayAttr(
      llvm::SmallVectorImpl<llvm::SmallVector<int64_t>> &intData);

  /// Create a temporary using `fir.alloca`. This function does not hoist.
  /// It is the callers responsibility to set the insertion point if
  /// hoisting is required.
  mlir::Value createTemporaryAlloc(
      mlir::Location loc, mlir::Type type, llvm::StringRef name,
      mlir::ValueRange lenParams = {}, mlir::ValueRange shape = {},
      llvm::ArrayRef<mlir::NamedAttribute> attrs = {},
      std::optional<Fortran::common::CUDADataAttr> cudaAttr = std::nullopt);

  /// Create a temporary. A temp is allocated using `fir.alloca` and can be read
  /// and written using `fir.load` and `fir.store`, resp.  The temporary can be
  /// given a name via a front-end `Symbol` or a `StringRef`.
  mlir::Value createTemporary(
      mlir::Location loc, mlir::Type type, llvm::StringRef name = {},
      mlir::ValueRange shape = {}, mlir::ValueRange lenParams = {},
      llvm::ArrayRef<mlir::NamedAttribute> attrs = {},
      std::optional<Fortran::common::CUDADataAttr> cudaAttr = std::nullopt);

  /// Create an unnamed and untracked temporary on the stack.
  mlir::Value createTemporary(mlir::Location loc, mlir::Type type,
                              mlir::ValueRange shape) {
    return createTemporary(loc, type, llvm::StringRef{}, shape);
  }

  mlir::Value createTemporary(mlir::Location loc, mlir::Type type,
                              llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    return createTemporary(loc, type, llvm::StringRef{}, {}, {}, attrs);
  }

  mlir::Value createTemporary(mlir::Location loc, mlir::Type type,
                              llvm::StringRef name,
                              llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    return createTemporary(loc, type, name, {}, {}, attrs);
  }

  /// Create a temporary on the heap.
  mlir::Value
  createHeapTemporary(mlir::Location loc, mlir::Type type,
                      llvm::StringRef name = {}, mlir::ValueRange shape = {},
                      mlir::ValueRange lenParams = {},
                      llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

  /// Sample genDeclare callback for createArrayTemp() below.
  /// It creates fir.declare operation using the given operands.
  /// \p memref is the base of the allocated temporary,
  /// which may be !fir.ref<!fir.array<>> or !fir.box/class<>.
  static mlir::Value genTempDeclareOp(fir::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Value memref,
                                      llvm::StringRef name, mlir::Value shape,
                                      llvm::ArrayRef<mlir::Value> typeParams,
                                      fir::FortranVariableFlagsAttr attrs);

  /// Create a temporary with the given \p baseType,
  /// \p shape, \p extents and \p typeParams. An optional
  /// \p polymorphicMold specifies the entity which dynamic type
  /// has to be used for the allocation.
  /// \p genDeclare callback generates a declare operation
  /// for the created temporary. FIR passes may use genTempDeclareOp()
  /// function above that creates fir.declare.
  /// HLFIR passes may provide their own callback that generates
  /// hlfir.declare. Some passes may provide a callback that
  /// just passes through the base of the temporary.
  /// If \p useStack is true, the function will try to do the allocation
  /// in stack memory (which is not always possible currently).
  /// The first return value is the base of the temporary object,
  /// which may be !fir.ref<!fir.array<>> or !fir.box/class<>.
  /// The second return value is true, if the actual allocation
  /// was done in heap memory.
  std::pair<mlir::Value, bool> createAndDeclareTemp(
      mlir::Location loc, mlir::Type baseType, mlir::Value shape,
      llvm::ArrayRef<mlir::Value> extents,
      llvm::ArrayRef<mlir::Value> typeParams,
      const std::function<decltype(genTempDeclareOp)> &genDeclare,
      mlir::Value polymorphicMold, bool useStack, llvm::StringRef tmpName);
  /// Create and declare an array temporary.
  std::pair<mlir::Value, bool>
  createArrayTemp(mlir::Location loc, fir::SequenceType arrayType,
                  mlir::Value shape, llvm::ArrayRef<mlir::Value> extents,
                  llvm::ArrayRef<mlir::Value> typeParams,
                  const std::function<decltype(genTempDeclareOp)> &genDeclare,
                  mlir::Value polymorphicMold, bool useStack = false,
                  llvm::StringRef tmpName = ".tmp.array") {
    return createAndDeclareTemp(loc, arrayType, shape, extents, typeParams,
                                genDeclare, polymorphicMold, useStack, tmpName);
  }

  /// Create an LLVM stack save intrinsic op. Returns the saved stack pointer.
  /// The stack address space is fetched from the data layout of the current
  /// module.
  mlir::Value genStackSave(mlir::Location loc);

  /// Create an LLVM stack restore intrinsic op. stackPointer should be a value
  /// previously returned from genStackSave.
  void genStackRestore(mlir::Location loc, mlir::Value stackPointer);

  /// Create a global value.
  fir::GlobalOp createGlobal(mlir::Location loc, mlir::Type type,
                             llvm::StringRef name,
                             mlir::StringAttr linkage = {},
                             mlir::Attribute value = {}, bool isConst = false,
                             bool isTarget = false,
                             cuf::DataAttributeAttr dataAttr = {});

  fir::GlobalOp createGlobal(mlir::Location loc, mlir::Type type,
                             llvm::StringRef name, bool isConst, bool isTarget,
                             std::function<void(FirOpBuilder &)> bodyBuilder,
                             mlir::StringAttr linkage = {},
                             cuf::DataAttributeAttr dataAttr = {});

  /// Create a global constant (read-only) value.
  fir::GlobalOp createGlobalConstant(mlir::Location loc, mlir::Type type,
                                     llvm::StringRef name,
                                     mlir::StringAttr linkage = {},
                                     mlir::Attribute value = {}) {
    return createGlobal(loc, type, name, linkage, value, /*isConst=*/true,
                        /*isTarget=*/false);
  }

  fir::GlobalOp
  createGlobalConstant(mlir::Location loc, mlir::Type type,
                       llvm::StringRef name,
                       std::function<void(FirOpBuilder &)> bodyBuilder,
                       mlir::StringAttr linkage = {}) {
    return createGlobal(loc, type, name, /*isConst=*/true, /*isTarget=*/false,
                        bodyBuilder, linkage);
  }

  /// Convert a StringRef string into a fir::StringLitOp.
  fir::StringLitOp createStringLitOp(mlir::Location loc,
                                     llvm::StringRef string);

  std::pair<fir::TypeInfoOp, mlir::OpBuilder::InsertPoint>
  createTypeInfoOp(mlir::Location loc, fir::RecordType recordType,
                   fir::RecordType parentType);

  //===--------------------------------------------------------------------===//
  // Linkage helpers (inline). The default linkage is external.
  //===--------------------------------------------------------------------===//

  static mlir::StringAttr createCommonLinkage(mlir::MLIRContext *context) {
    return mlir::StringAttr::get(context, "common");
  }
  mlir::StringAttr createCommonLinkage() {
    return createCommonLinkage(getContext());
  }

  mlir::StringAttr createExternalLinkage() { return getStringAttr("external"); }

  mlir::StringAttr createInternalLinkage() { return getStringAttr("internal"); }

  mlir::StringAttr createLinkOnceLinkage() { return getStringAttr("linkonce"); }

  mlir::StringAttr createLinkOnceODRLinkage() {
    return getStringAttr("linkonce_odr");
  }

  mlir::StringAttr createWeakLinkage() { return getStringAttr("weak"); }

  /// Get a function by name. If the function exists in the current module, it
  /// is returned. Otherwise, a null FuncOp is returned.
  mlir::func::FuncOp getNamedFunction(llvm::StringRef name) {
    return getNamedFunction(getModule(), getMLIRSymbolTable(), name);
  }
  static mlir::func::FuncOp
  getNamedFunction(mlir::ModuleOp module, const mlir::SymbolTable *symbolTable,
                   llvm::StringRef name);

  /// Get a function by symbol name. The result will be null if there is no
  /// function with the given symbol in the module.
  mlir::func::FuncOp getNamedFunction(mlir::SymbolRefAttr symbol) {
    return getNamedFunction(getModule(), getMLIRSymbolTable(), symbol);
  }
  static mlir::func::FuncOp
  getNamedFunction(mlir::ModuleOp module, const mlir::SymbolTable *symbolTable,
                   mlir::SymbolRefAttr symbol);

  fir::GlobalOp getNamedGlobal(llvm::StringRef name) {
    return getNamedGlobal(getModule(), getMLIRSymbolTable(), name);
  }

  static fir::GlobalOp getNamedGlobal(mlir::ModuleOp module,
                                      const mlir::SymbolTable *symbolTable,
                                      llvm::StringRef name);

  /// Lazy creation of fir.convert op.
  mlir::Value createConvert(mlir::Location loc, mlir::Type toTy,
                            mlir::Value val);

  /// Create a fir.convert op with a volatile cast if the source value's type
  /// does not match the target type's volatility.
  mlir::Value createConvertWithVolatileCast(mlir::Location loc, mlir::Type toTy,
                                            mlir::Value val);

  /// Cast \p value to have \p isVolatile volatility.
  mlir::Value createVolatileCast(mlir::Location loc, bool isVolatile,
                                 mlir::Value value);

  /// Create a fir.store of \p val into \p addr. A lazy conversion
  /// of \p val to the element type of \p addr is created if needed.
  void createStoreWithConvert(mlir::Location loc, mlir::Value val,
                              mlir::Value addr);

  /// Create a fir.load if \p val is a reference or pointer type. Return the
  /// result of the load if it was created, otherwise return \p val
  mlir::Value loadIfRef(mlir::Location loc, mlir::Value val);

  /// Determine if the named function is already in the module. Return the
  /// instance if found, otherwise add a new named function to the module.
  mlir::func::FuncOp createFunction(mlir::Location loc, llvm::StringRef name,
                                    mlir::FunctionType ty) {
    return createFunction(loc, getModule(), name, ty, getMLIRSymbolTable());
  }

  static mlir::func::FuncOp createFunction(mlir::Location loc,
                                           mlir::ModuleOp module,
                                           llvm::StringRef name,
                                           mlir::FunctionType ty,
                                           mlir::SymbolTable *);

  /// Returns a named function for a Fortran runtime API, creating
  /// it, if it does not exist in the module yet.
  /// If \p isIO is set to true, then the function corresponds
  /// to one of Fortran runtime IO APIs.
  mlir::func::FuncOp createRuntimeFunction(mlir::Location loc,
                                           llvm::StringRef name,
                                           mlir::FunctionType ty,
                                           bool isIO = false);

  /// Cast the input value to IndexType.
  mlir::Value convertToIndexType(mlir::Location loc, mlir::Value val) {
    return createConvert(loc, getIndexType(), val);
  }

  /// Construct one of the two forms of shape op from an array box.
  mlir::Value genShape(mlir::Location loc, const fir::AbstractArrayBox &arr);
  mlir::Value genShape(mlir::Location loc, llvm::ArrayRef<mlir::Value> shift,
                       llvm::ArrayRef<mlir::Value> exts);
  mlir::Value genShape(mlir::Location loc, llvm::ArrayRef<mlir::Value> exts);
  mlir::Value genShift(mlir::Location loc, llvm::ArrayRef<mlir::Value> shift);

  /// Create one of the shape ops given an extended value. For a boxed value,
  /// this may create a `fir.shift` op.
  mlir::Value createShape(mlir::Location loc, const fir::ExtendedValue &exv);

  /// Create a slice op extended value. The value to be sliced, `exv`, must be
  /// an array.
  mlir::Value createSlice(mlir::Location loc, const fir::ExtendedValue &exv,
                          mlir::ValueRange triples, mlir::ValueRange path);

  /// Create a boxed value (Fortran descriptor) to be passed to the runtime.
  /// \p exv is an extended value holding a memory reference to the object that
  /// must be boxed. This function will crash if provided something that is not
  /// a memory reference type.
  /// Array entities are boxed with a shape and possibly a shift. Character
  /// entities are boxed with a LEN parameter.
  mlir::Value createBox(mlir::Location loc, const fir::ExtendedValue &exv,
                        bool isPolymorphic = false, bool isAssumedType = false);

  mlir::Value createBox(mlir::Location loc, mlir::Type boxType,
                        mlir::Value addr, mlir::Value shape, mlir::Value slice,
                        llvm::ArrayRef<mlir::Value> lengths, mlir::Value tdesc);

  /// Create constant i1 with value 1. if \p b is true or 0. otherwise
  mlir::Value createBool(mlir::Location loc, bool b) {
    return createIntegerConstant(loc, getIntegerType(1), b ? 1 : 0);
  }

  //===--------------------------------------------------------------------===//
  // If-Then-Else generation helper
  //===--------------------------------------------------------------------===//

  /// Helper class to create if-then-else in a structured way:
  /// Usage: genIfOp().genThen([&](){...}).genElse([&](){...}).end();
  /// Alternatively, getResults() can be used instead of end() to end the ifOp
  /// and get the ifOp results.
  class IfBuilder {
  public:
    IfBuilder(fir::IfOp ifOp, FirOpBuilder &builder)
        : ifOp{ifOp}, builder{builder} {}
    template <typename CC>
    IfBuilder &genThen(CC func) {
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      func();
      return *this;
    }
    template <typename CC>
    IfBuilder &genElse(CC func) {
      assert(!ifOp.getElseRegion().empty() && "must have else region");
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      func();
      return *this;
    }
    void end() { builder.setInsertionPointAfter(ifOp); }

    /// End the IfOp and return the results if any.
    mlir::Operation::result_range getResults() {
      end();
      return ifOp.getResults();
    }

    fir::IfOp &getIfOp() { return ifOp; };

  private:
    fir::IfOp ifOp;
    FirOpBuilder &builder;
  };

  /// Create an IfOp and returns an IfBuilder that can generate the else/then
  /// bodies.
  IfBuilder genIfOp(mlir::Location loc, mlir::TypeRange results,
                    mlir::Value cdt, bool withElseRegion) {
    auto op = fir::IfOp::create(*this, loc, results, cdt, withElseRegion);
    return IfBuilder(op, *this);
  }

  /// Create an IfOp with no "else" region, and no result values.
  /// Usage: genIfThen(loc, cdt).genThen(lambda).end();
  IfBuilder genIfThen(mlir::Location loc, mlir::Value cdt) {
    auto op = fir::IfOp::create(*this, loc, mlir::TypeRange(), cdt, false);
    return IfBuilder(op, *this);
  }

  /// Create an IfOp with an "else" region, and no result values.
  /// Usage: genIfThenElse(loc, cdt).genThen(lambda).genElse(lambda).end();
  IfBuilder genIfThenElse(mlir::Location loc, mlir::Value cdt) {
    auto op = fir::IfOp::create(*this, loc, mlir::TypeRange(), cdt, true);
    return IfBuilder(op, *this);
  }

  mlir::Value genNot(mlir::Location loc, mlir::Value boolean) {
    return mlir::arith::CmpIOp::create(*this, loc,
                                       mlir::arith::CmpIPredicate::eq, boolean,
                                       createBool(loc, false));
  }

  /// Generate code testing \p addr is not a null address.
  mlir::Value genIsNotNullAddr(mlir::Location loc, mlir::Value addr);

  /// Generate code testing \p addr is a null address.
  mlir::Value genIsNullAddr(mlir::Location loc, mlir::Value addr);

  /// Compute the extent of (lb:ub:step) as max((ub-lb+step)/step, 0). See
  /// Fortran 2018 9.5.3.3.2 section for more details.
  mlir::Value genExtentFromTriplet(mlir::Location loc, mlir::Value lb,
                                   mlir::Value ub, mlir::Value step,
                                   mlir::Type type);

  /// Create an AbsentOp of \p argTy type and handle special cases, such as
  /// Character Procedure Tuple arguments.
  mlir::Value genAbsentOp(mlir::Location loc, mlir::Type argTy);

  /// Set default FastMathFlags value for all operations
  /// supporting mlir::arith::FastMathAttr that will be created
  /// by this builder.
  void setFastMathFlags(mlir::arith::FastMathFlags flags) {
    fastMathFlags = flags;
  }

  /// Set default FastMathFlags value from the passed MathOptionsBase
  /// config.
  void setFastMathFlags(Fortran::common::MathOptionsBase options);

  /// Get current FastMathFlags value.
  mlir::arith::FastMathFlags getFastMathFlags() const { return fastMathFlags; }

  /// Stringify FastMathFlags set in a way
  /// that the string may be used for mangling a function name.
  /// If FastMathFlags are set to 'none', then the result is an empty
  /// string.
  std::string getFastMathFlagsString() {
    mlir::arith::FastMathFlags flags = getFastMathFlags();
    if (flags == mlir::arith::FastMathFlags::none)
      return {};

    std::string fmfString{mlir::arith::stringifyFastMathFlags(flags)};
    std::replace(fmfString.begin(), fmfString.end(), ',', '_');
    return fmfString;
  }

  /// Set default IntegerOverflowFlags value for all operations
  /// supporting mlir::arith::IntegerOverflowFlagsAttr that will be created
  /// by this builder.
  void setIntegerOverflowFlags(mlir::arith::IntegerOverflowFlags flags) {
    integerOverflowFlags = flags;
  }

  /// Get current IntegerOverflowFlags value.
  mlir::arith::IntegerOverflowFlags getIntegerOverflowFlags() const {
    return integerOverflowFlags;
  }

  /// Set ComplexDivisionToRuntimeFlag value. If set to true, complex number
  /// division is lowered to a runtime function by this builder.
  void setComplexDivisionToRuntimeFlag(bool flag) {
    complexDivisionToRuntimeFlag = flag;
  }

  /// Get current ComplexDivisionToRuntimeFlag value.
  bool getComplexDivisionToRuntimeFlag() const {
    return complexDivisionToRuntimeFlag;
  }

  /// Dump the current function. (debug)
  LLVM_DUMP_METHOD void dumpFunc();

  /// FirOpBuilder hook for creating new operation.
  void notifyOperationInserted(mlir::Operation *op,
                               mlir::OpBuilder::InsertPoint previous) override {
    // We only care about newly created operations.
    if (previous.isSet())
      return;
    setCommonAttributes(op);
  }

  /// Construct a data layout on demand and return it
  mlir::DataLayout &getDataLayout();

  /// Convert operands &/or result from/to unsigned so that the operation
  /// only receives/produces signless operands.
  template <typename OpTy>
  mlir::Value createUnsigned(mlir::Location loc, mlir::Type resultType,
                             mlir::Value left, mlir::Value right) {
    if (!resultType.isIntOrFloat())
      return OpTy::create(*this, loc, resultType, left, right);
    mlir::Type signlessType = mlir::IntegerType::get(
        getContext(), resultType.getIntOrFloatBitWidth(),
        mlir::IntegerType::SignednessSemantics::Signless);
    mlir::Type opResType = resultType;
    if (left.getType().isUnsignedInteger()) {
      left = createConvert(loc, signlessType, left);
      opResType = signlessType;
    }
    if (right.getType().isUnsignedInteger()) {
      right = createConvert(loc, signlessType, right);
      opResType = signlessType;
    }
    mlir::Value result = OpTy::create(*this, loc, opResType, left, right);
    if (resultType.isUnsignedInteger())
      result = createConvert(loc, resultType, result);
    return result;
  }

  /// Compare two pointer-like values using the given predicate.
  mlir::Value genPtrCompare(mlir::Location loc,
                            mlir::arith::CmpIPredicate predicate,
                            mlir::Value ptr1, mlir::Value ptr2) {
    ptr1 = createConvert(loc, getIndexType(), ptr1);
    ptr2 = createConvert(loc, getIndexType(), ptr2);
    return mlir::arith::CmpIOp::create(*this, loc, predicate, ptr1, ptr2);
  }

private:
  /// Set attributes (e.g. FastMathAttr) to \p op operation
  /// based on the current attributes setting.
  void setCommonAttributes(mlir::Operation *op) const;

  KindMapping kindMap;

  /// FastMathFlags that need to be set for operations that support
  /// mlir::arith::FastMathAttr.
  mlir::arith::FastMathFlags fastMathFlags{};

  /// IntegerOverflowFlags that need to be set for operations that support
  /// mlir::arith::IntegerOverflowFlagsAttr.
  mlir::arith::IntegerOverflowFlags integerOverflowFlags{};

  /// Flag to control whether complex number division is lowered to a runtime
  /// function or to the MLIR complex dialect.
  bool complexDivisionToRuntimeFlag = true;

  /// fir::GlobalOp and func::FuncOp symbol table to speed-up
  /// lookups.
  mlir::SymbolTable *symbolTable = nullptr;

  /// DataLayout constructed on demand. Access via getDataLayout().
  /// Stored via a unique_ptr rather than an optional so as not to bloat this
  /// class when most instances won't ever need a data layout.
  std::unique_ptr<mlir::DataLayout> dataLayout = nullptr;
};

} // namespace fir

namespace fir::factory {

//===----------------------------------------------------------------------===//
// ExtendedValue inquiry helpers
//===----------------------------------------------------------------------===//

/// Read or get character length from \p box that must contain a character
/// entity. If the length value is contained in the ExtendedValue, this will
/// not generate any code, otherwise this will generate a read of the fir.box
/// describing the entity.
mlir::Value readCharLen(fir::FirOpBuilder &builder, mlir::Location loc,
                        const fir::ExtendedValue &box);

/// Read or get the extent in dimension \p dim of the array described by \p box.
mlir::Value readExtent(fir::FirOpBuilder &builder, mlir::Location loc,
                       const fir::ExtendedValue &box, unsigned dim);

/// Read or get the lower bound in dimension \p dim of the array described by
/// \p box. If the lower bound is left default in the ExtendedValue,
/// \p defaultValue will be returned.
mlir::Value readLowerBound(fir::FirOpBuilder &builder, mlir::Location loc,
                           const fir::ExtendedValue &box, unsigned dim,
                           mlir::Value defaultValue);

/// Read extents from \p box.
llvm::SmallVector<mlir::Value> readExtents(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           const fir::BoxValue &box);

/// Read a fir::BoxValue into an fir::UnboxValue, a fir::ArrayBoxValue or a
/// fir::CharArrayBoxValue. This should only be called if the fir::BoxValue is
/// known to be contiguous given the context (or if the resulting address will
/// not be used). If the value is polymorphic, its dynamic type will be lost.
/// This must not be used on unlimited polymorphic and assumed rank entities.
fir::ExtendedValue readBoxValue(fir::FirOpBuilder &builder, mlir::Location loc,
                                const fir::BoxValue &box);

/// Get the lower bounds of \p exv. NB: returns an empty vector if the lower
/// bounds are all ones, which is the default in Fortran.
llvm::SmallVector<mlir::Value>
getNonDefaultLowerBounds(fir::FirOpBuilder &builder, mlir::Location loc,
                         const fir::ExtendedValue &exv);

/// Return LEN parameters associated to \p exv that are not deferred (that are
/// available without having to read any fir.box values). Empty if \p exv has no
/// LEN parameters or if they are all deferred.
llvm::SmallVector<mlir::Value>
getNonDeferredLenParams(const fir::ExtendedValue &exv);

//===----------------------------------------------------------------------===//
// String literal helper helpers
//===----------------------------------------------------------------------===//

/// Create a !fir.char<1> string literal global and returns a fir::CharBoxValue
/// with its address and length.
fir::ExtendedValue createStringLiteral(fir::FirOpBuilder &, mlir::Location,
                                       llvm::StringRef string);

/// Unique a compiler generated identifier. A short prefix should be provided
/// to hint at the origin of the identifier.
std::string uniqueCGIdent(llvm::StringRef prefix, llvm::StringRef name);

/// Lowers the extents from the sequence type to Values.
/// Any unknown extents are lowered to undefined values.
llvm::SmallVector<mlir::Value> createExtents(fir::FirOpBuilder &builder,
                                             mlir::Location loc,
                                             fir::SequenceType seqTy);

//===--------------------------------------------------------------------===//
// Location helpers
//===--------------------------------------------------------------------===//

/// Generate a string literal containing the file name and return its address
mlir::Value locationToFilename(fir::FirOpBuilder &, mlir::Location);
/// Generate a constant of the given type with the location line number
mlir::Value locationToLineNo(fir::FirOpBuilder &, mlir::Location, mlir::Type);

//===--------------------------------------------------------------------===//
// ExtendedValue helpers
//===--------------------------------------------------------------------===//

/// Return the extended value for a component of a derived type instance given
/// the address of the component.
fir::ExtendedValue componentToExtendedValue(fir::FirOpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value component);

/// Given the address of an array element and the ExtendedValue describing the
/// array, returns the ExtendedValue describing the array element. The purpose
/// is to propagate the LEN parameters of the array to the element. This can be
/// used for elements of `array` or `array(i:j:k)`. If \p element belongs to an
/// array section `array%x` whose base is \p array,
/// arraySectionElementToExtendedValue must be used instead.
fir::ExtendedValue arrayElementToExtendedValue(fir::FirOpBuilder &builder,
                                               mlir::Location loc,
                                               const fir::ExtendedValue &array,
                                               mlir::Value element);

/// Build the ExtendedValue for \p element that is an element of an array or
/// array section with \p array base (`array` or `array(i:j:k)%x%y`).
/// If it is an array section, \p slice must be provided and be a fir::SliceOp
/// that describes the section.
fir::ExtendedValue arraySectionElementToExtendedValue(
    fir::FirOpBuilder &builder, mlir::Location loc,
    const fir::ExtendedValue &array, mlir::Value element, mlir::Value slice);

/// Assign \p rhs to \p lhs. Both \p rhs and \p lhs must be scalars. The
/// assignment follows Fortran intrinsic assignment semantic (10.2.1.3).
void genScalarAssignment(fir::FirOpBuilder &builder, mlir::Location loc,
                         const fir::ExtendedValue &lhs,
                         const fir::ExtendedValue &rhs,
                         bool needFinalization = false,
                         bool isTemporaryLHS = false);

/// Assign \p rhs to \p lhs. Both \p rhs and \p lhs must be scalar derived
/// types. The assignment follows Fortran intrinsic assignment semantic for
/// derived types (10.2.1.3 point 13).
void genRecordAssignment(fir::FirOpBuilder &builder, mlir::Location loc,
                         const fir::ExtendedValue &lhs,
                         const fir::ExtendedValue &rhs,
                         bool needFinalization = false,
                         bool isTemporaryLHS = false);

/// Builds and returns the type of a ragged array header used to cache mask
/// evaluations. RaggedArrayHeader is defined in
/// flang/include/flang/Runtime/ragged.h.
mlir::TupleType getRaggedArrayHeaderType(fir::FirOpBuilder &builder);

/// Generate the, possibly dynamic, LEN of a CHARACTER. \p arrLoad determines
/// the base array. After applying \p path, the result must be a reference to a
/// `!fir.char` type object. \p substring must have 0, 1, or 2 members. The
/// first member is the starting offset. The second is the ending offset.
mlir::Value genLenOfCharacter(fir::FirOpBuilder &builder, mlir::Location loc,
                              fir::ArrayLoadOp arrLoad,
                              llvm::ArrayRef<mlir::Value> path,
                              llvm::ArrayRef<mlir::Value> substring);
mlir::Value genLenOfCharacter(fir::FirOpBuilder &builder, mlir::Location loc,
                              fir::SequenceType seqTy, mlir::Value memref,
                              llvm::ArrayRef<mlir::Value> typeParams,
                              llvm::ArrayRef<mlir::Value> path,
                              llvm::ArrayRef<mlir::Value> substring);

/// Create the zero value of a given the numerical or logical \p type (`false`
/// for logical types).
mlir::Value createZeroValue(fir::FirOpBuilder &builder, mlir::Location loc,
                            mlir::Type type);

/// Get the integer constants of triplet and compute the extent.
std::optional<std::int64_t> getExtentFromTriplet(mlir::Value lb, mlir::Value ub,
                                                 mlir::Value stride);

/// Compute the extent value given the lower bound \lb and upper bound \ub.
/// All inputs must have the same SSA integer type.
mlir::Value computeExtent(fir::FirOpBuilder &builder, mlir::Location loc,
                          mlir::Value lb, mlir::Value ub);
mlir::Value computeExtent(fir::FirOpBuilder &builder, mlir::Location loc,
                          mlir::Value lb, mlir::Value ub, mlir::Value zero,
                          mlir::Value one);

/// Generate max(\p value, 0) where \p value is a scalar integer.
mlir::Value genMaxWithZero(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value value);
mlir::Value genMaxWithZero(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value value, mlir::Value zero);

/// The type(C_PTR/C_FUNPTR) is defined as the derived type with only one
/// component of integer 64, and the component is the C address. Get the C
/// address.
mlir::Value genCPtrOrCFunptrAddr(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value cPtr, mlir::Type ty);

/// The type(C_DEVPTR) is defined as the derived type with only one
/// component of C_PTR type. Get the C address from the C_PTR component.
mlir::Value genCDevPtrAddr(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value cDevPtr, mlir::Type ty);

/// Get the C address value.
mlir::Value genCPtrOrCFunptrValue(fir::FirOpBuilder &builder,
                                  mlir::Location loc, mlir::Value cPtr);

/// Create a fir.box from a fir::ExtendedValue and wrap it in a fir::BoxValue
/// to keep all the lower bound and explicit parameter information.
fir::BoxValue createBoxValue(fir::FirOpBuilder &builder, mlir::Location loc,
                             const fir::ExtendedValue &exv);

/// Generate Null BoxProc for procedure pointer null initialization.
mlir::Value createNullBoxProc(fir::FirOpBuilder &builder, mlir::Location loc,
                              mlir::Type boxType);

/// Convert a value to a new type. Return the value directly if it has the right
/// type.
mlir::Value createConvert(mlir::OpBuilder &, mlir::Location, mlir::Type,
                          mlir::Value);

/// Set internal linkage attribute on a function.
void setInternalLinkage(mlir::func::FuncOp);

llvm::SmallVector<mlir::Value>
elideExtentsAlreadyInType(mlir::Type type, mlir::ValueRange shape);

llvm::SmallVector<mlir::Value>
elideLengthsAlreadyInType(mlir::Type type, mlir::ValueRange lenParams);

/// Get the address space which should be used for allocas
uint64_t getAllocaAddressSpace(const mlir::DataLayout *dataLayout);

/// The two vectors of MLIR values have the following property:
///   \p extents1[i] must have the same value as \p extents2[i]
/// The function returns a new vector of MLIR values that preserves
/// the same property vs \p extents1 and \p extents2, but allows
/// more optimizations. For example, if extents1[j] is a known constant,
/// and extents2[j] is not, then result[j] is the MLIR value extents1[j].
llvm::SmallVector<mlir::Value> deduceOptimalExtents(mlir::ValueRange extents1,
                                                    mlir::ValueRange extents2);

uint64_t getGlobalAddressSpace(mlir::DataLayout *dataLayout);

uint64_t getProgramAddressSpace(mlir::DataLayout *dataLayout);

/// Given array extents generate code that sets them all to zeroes,
/// if the array is empty, e.g.:
///   %false = arith.constant false
///   %c0 = arith.constant 0 : index
///   %p1 = arith.cmpi eq, %e0, %c0 : index
///   %p2 = arith.ori %false, %p1 : i1
///   %p3 = arith.cmpi eq, %e1, %c0 : index
///   %p4 = arith.ori %p1, %p2 : i1
///   %result0 = arith.select %p4, %c0, %e0 : index
///   %result1 = arith.select %p4, %c0, %e1 : index
llvm::SmallVector<mlir::Value> updateRuntimeExtentsForEmptyArrays(
    fir::FirOpBuilder &builder, mlir::Location loc, mlir::ValueRange extents);

/// Given \p box of type fir::BaseBoxType representing an array,
/// the function generates code to fetch the lower bounds,
/// the extents and the strides from the box. The values are returned via
/// \p lbounds, \p extents and \p strides.
void genDimInfoFromBox(fir::FirOpBuilder &builder, mlir::Location loc,
                       mlir::Value box,
                       llvm::SmallVectorImpl<mlir::Value> *lbounds,
                       llvm::SmallVectorImpl<mlir::Value> *extents,
                       llvm::SmallVectorImpl<mlir::Value> *strides);

/// Generate an LLVM dialect lifetime start marker at the current insertion
/// point given an fir.alloca. Returns the value to be passed to the lifetime
/// end marker.
mlir::Value genLifetimeStart(mlir::OpBuilder &builder, mlir::Location loc,
                             fir::AllocaOp alloc, const mlir::DataLayout *dl);

/// Generate an LLVM dialect lifetime end marker at the current insertion point
/// given an llvm.ptr value.
void genLifetimeEnd(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value mem);

/// Given a fir.box or fir.class \p box describing an entity and a raw address
/// \p newAddr for an entity with the same Fortran properties (rank, dynamic
/// type, length parameters and bounds) and attributes (POINTER or ALLOCATABLE),
/// create a box for \p newAddr with the same type as \p box. This assumes \p
/// newAddr is for contiguous storage (\p box does not have to be contiguous).
mlir::Value getDescriptorWithNewBaseAddress(fir::FirOpBuilder &builder,
                                            mlir::Location loc, mlir::Value box,
                                            mlir::Value newAddr);

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_FIRBUILDER_H
