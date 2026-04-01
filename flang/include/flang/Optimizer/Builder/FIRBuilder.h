//===-- FirBuilder.h -- FIR operation builder -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of AIIR. As FIR is a
// dialect of AIIR, it makes extensive use of AIIR interfaces and AIIR's coding
// style (https://aiir.llvm.org/getting_started/DeveloperGuide/) is used in this
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
#include "flang/Support/FPMaxminBehavior.h"
#include "flang/Support/MathOptionsBase.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>
#include <utility>

namespace aiir {
class DataLayout;
class SymbolTable;
}

namespace fir {
class AbstractArrayBox;
class ExtendedValue;
class MutableBoxValue;
class BoxValue;

/// Get the integer type with a pointer size.
inline aiir::Type getIntPtrType(aiir::OpBuilder &builder) {
  // TODO: Delay the need of such type until codegen or find a way to use
  // llvm::DataLayout::getPointerSizeInBits here.
  return builder.getI64Type();
}

//===----------------------------------------------------------------------===//
// FirOpBuilder
//===----------------------------------------------------------------------===//

/// Extends the AIIR OpBuilder to provide methods for building common FIR
/// patterns.
class FirOpBuilder : public aiir::OpBuilder, public aiir::OpBuilder::Listener {
public:
  explicit FirOpBuilder(aiir::Operation *op, fir::KindMapping kindMap,
                        aiir::SymbolTable *symbolTable = nullptr)
      : OpBuilder{op, /*listener=*/this}, kindMap{std::move(kindMap)},
        symbolTable{symbolTable} {
    auto fmi = aiir::dyn_cast<aiir::arith::ArithFastMathInterface>(*op);
    if (fmi) {
      // Set the builder with FastMathFlags attached to the operation.
      setFastMathFlags(fmi.getFastMathFlagsAttr().getValue());
    }
  }
  explicit FirOpBuilder(aiir::OpBuilder &builder, fir::KindMapping kindMap,
                        aiir::SymbolTable *symbolTable = nullptr)
      : OpBuilder(builder), OpBuilder::Listener(), kindMap{std::move(kindMap)},
        symbolTable{symbolTable} {
    setListener(this);
  }
  explicit FirOpBuilder(aiir::OpBuilder &builder, aiir::ModuleOp mod)
      : OpBuilder(builder), OpBuilder::Listener(),
        kindMap{getKindMapping(mod)} {
    setListener(this);
  }
  explicit FirOpBuilder(aiir::OpBuilder &builder, fir::KindMapping kindMap,
                        aiir::Operation *op)
      : OpBuilder(builder), OpBuilder::Listener(), kindMap{std::move(kindMap)} {
    setListener(this);
    auto fmi = aiir::dyn_cast<aiir::arith::ArithFastMathInterface>(*op);
    if (fmi) {
      // Set the builder with FastMathFlags attached to the operation.
      setFastMathFlags(fmi.getFastMathFlagsAttr().getValue());
    }
  }
  FirOpBuilder(aiir::OpBuilder &builder, aiir::Operation *op)
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
  aiir::Region &getRegion() { return *getBlock()->getParent(); }

  /// Get the current Module
  aiir::ModuleOp getModule() {
    return getRegion().getParentOfType<aiir::ModuleOp>();
  }

  /// Get the current Function
  aiir::func::FuncOp getFunction() {
    return getRegion().getParentOfType<aiir::func::FuncOp>();
  }

  /// Get a reference to the kind map.
  const fir::KindMapping &getKindMap() { return kindMap; }

  /// Get func.func/fir.global symbol table attached to this builder if any.
  aiir::SymbolTable *getAIIRSymbolTable() { return symbolTable; }

  /// Get the default integer type
  [[maybe_unused]] aiir::IntegerType getDefaultIntegerType() {
    return getIntegerType(
        getKindMap().getIntegerBitsize(getKindMap().defaultIntegerKind()));
  }

  /// The LHS and RHS are not always in agreement in terms of type. In some
  /// cases, the disagreement is between COMPLEX and other scalar types. In that
  /// case, the conversion must insert (extract) out of a COMPLEX value to have
  /// the proper semantics and be strongly typed. E.g., converting an integer
  /// (real) to a complex, the real part is filled using the integer (real)
  /// after type conversion and the imaginary part is zero.
  aiir::Value convertWithSemantics(aiir::Location loc, aiir::Type toTy,
                                   aiir::Value val,
                                   bool allowCharacterConversion = false,
                                   bool allowRebox = false);

  /// Get the entry block of the current Function
  aiir::Block *getEntryBlock() { return &getFunction().front(); }

  /// Get the block for adding Allocas. If OpenMP is enabled then get the
  /// the alloca block from an Operation which can be Outlined. Otherwise
  /// use the entry block of the current Function
  aiir::Block *getAllocaBlock();

  /// Safely create a reference type to the type `eleTy`.
  aiir::Type getRefType(aiir::Type eleTy, bool isVolatile = false);

  /// Create a sequence of `eleTy` with `rank` dimensions of unknown size.
  aiir::Type getVarLenSeqTy(aiir::Type eleTy, unsigned rank = 1);

  /// Get character length type.
  aiir::Type getCharacterLengthType() { return getIndexType(); }

  /// Get the integer type whose bit width corresponds to the width of pointer
  /// types, or is bigger.
  aiir::Type getIntPtrType() { return fir::getIntPtrType(*this); }

  /// Wrap `str` to a SymbolRefAttr.
  aiir::SymbolRefAttr getSymbolRefAttr(llvm::StringRef str) {
    return aiir::SymbolRefAttr::get(getContext(), str);
  }

  /// Get the aiir float type that implements Fortran REAL(kind).
  aiir::Type getRealType(int kind);

  fir::BoxProcType getBoxProcType(aiir::FunctionType funcTy) {
    return fir::BoxProcType::get(getContext(), funcTy);
  }

  /// Create a null constant memory reference of type \p ptrType.
  /// If \p ptrType is not provided, !fir.ref<none> type will be used.
  aiir::Value createNullConstant(aiir::Location loc, aiir::Type ptrType = {});

  /// Create an integer constant of type \p type and value \p i.
  /// Should not be used with negative values with integer types of more
  /// than 64 bits.
  aiir::Value createIntegerConstant(aiir::Location loc, aiir::Type integerType,
                                    std::int64_t i);

  /// Create an integer of \p integerType where all the bits have been set to
  /// ones. Safe to use regardless of integerType bitwidth.
  aiir::Value createAllOnesInteger(aiir::Location loc, aiir::Type integerType);

  /// Create -1 constant of \p integerType. Safe to use regardless of
  /// integerType bitwidth.
  aiir::Value createMinusOneInteger(aiir::Location loc,
                                    aiir::Type integerType) {
    return createAllOnesInteger(loc, integerType);
  }

  /// Create a real constant from an integer value.
  aiir::Value createRealConstant(aiir::Location loc, aiir::Type realType,
                                 llvm::APFloat::integerPart val);

  /// Create a real constant from an APFloat value.
  aiir::Value createRealConstant(aiir::Location loc, aiir::Type realType,
                                 const llvm::APFloat &val);

  /// Create a real constant of type \p realType with a value zero.
  aiir::Value createRealZeroConstant(aiir::Location loc, aiir::Type realType) {
    return createRealConstant(loc, realType, 0u);
  }

  /// Create a real constant of type \p realType with value one.
  aiir::Value createRealOneConstant(aiir::Location loc, aiir::Type realType) {
    return createRealConstant(loc, realType, 1u);
  }

  /// Create a slot for a local on the stack. Besides the variable's type and
  /// shape, it may be given name, pinned, or target attributes.
  aiir::Value allocateLocal(aiir::Location loc, aiir::Type ty,
                            llvm::StringRef uniqName, llvm::StringRef name,
                            bool pinned, llvm::ArrayRef<aiir::Value> shape,
                            llvm::ArrayRef<aiir::Value> lenParams,
                            bool asTarget = false);
  aiir::Value allocateLocal(aiir::Location loc, aiir::Type ty,
                            llvm::StringRef uniqName, llvm::StringRef name,
                            llvm::ArrayRef<aiir::Value> shape,
                            llvm::ArrayRef<aiir::Value> lenParams,
                            bool asTarget = false);

  /// Create a two dimensional ArrayAttr containing integer data as
  /// IntegerAttrs, effectively: ArrayAttr<ArrayAttr<IntegerAttr>>>.
  aiir::ArrayAttr create2DI64ArrayAttr(
      llvm::SmallVectorImpl<llvm::SmallVector<int64_t>> &intData);

  /// Create a temporary using `fir.alloca`. This function does not hoist.
  /// It is the callers responsibility to set the insertion point if
  /// hoisting is required.
  aiir::Value createTemporaryAlloc(
      aiir::Location loc, aiir::Type type, llvm::StringRef name,
      aiir::ValueRange lenParams = {}, aiir::ValueRange shape = {},
      llvm::ArrayRef<aiir::NamedAttribute> attrs = {},
      std::optional<Fortran::common::CUDADataAttr> cudaAttr = std::nullopt);

  /// Create a temporary. A temp is allocated using `fir.alloca` and can be read
  /// and written using `fir.load` and `fir.store`, resp.  The temporary can be
  /// given a name via a front-end `Symbol` or a `StringRef`.
  aiir::Value createTemporary(
      aiir::Location loc, aiir::Type type, llvm::StringRef name = {},
      aiir::ValueRange shape = {}, aiir::ValueRange lenParams = {},
      llvm::ArrayRef<aiir::NamedAttribute> attrs = {},
      std::optional<Fortran::common::CUDADataAttr> cudaAttr = std::nullopt);

  /// Create an unnamed and untracked temporary on the stack.
  aiir::Value createTemporary(aiir::Location loc, aiir::Type type,
                              aiir::ValueRange shape) {
    return createTemporary(loc, type, llvm::StringRef{}, shape);
  }

  aiir::Value createTemporary(aiir::Location loc, aiir::Type type,
                              llvm::ArrayRef<aiir::NamedAttribute> attrs) {
    return createTemporary(loc, type, llvm::StringRef{}, {}, {}, attrs);
  }

  aiir::Value createTemporary(aiir::Location loc, aiir::Type type,
                              llvm::StringRef name,
                              llvm::ArrayRef<aiir::NamedAttribute> attrs) {
    return createTemporary(loc, type, name, {}, {}, attrs);
  }

  /// Create a temporary on the heap.
  aiir::Value
  createHeapTemporary(aiir::Location loc, aiir::Type type,
                      llvm::StringRef name = {}, aiir::ValueRange shape = {},
                      aiir::ValueRange lenParams = {},
                      llvm::ArrayRef<aiir::NamedAttribute> attrs = {});

  /// Sample genDeclare callback for createArrayTemp() below.
  /// It creates fir.declare operation using the given operands.
  /// \p memref is the base of the allocated temporary,
  /// which may be !fir.ref<!fir.array<>> or !fir.box/class<>.
  static aiir::Value genTempDeclareOp(fir::FirOpBuilder &builder,
                                      aiir::Location loc, aiir::Value memref,
                                      llvm::StringRef name, aiir::Value shape,
                                      llvm::ArrayRef<aiir::Value> typeParams,
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
  std::pair<aiir::Value, bool> createAndDeclareTemp(
      aiir::Location loc, aiir::Type baseType, aiir::Value shape,
      llvm::ArrayRef<aiir::Value> extents,
      llvm::ArrayRef<aiir::Value> typeParams,
      const std::function<decltype(genTempDeclareOp)> &genDeclare,
      aiir::Value polymorphicMold, bool useStack, llvm::StringRef tmpName);
  /// Create and declare an array temporary.
  std::pair<aiir::Value, bool>
  createArrayTemp(aiir::Location loc, fir::SequenceType arrayType,
                  aiir::Value shape, llvm::ArrayRef<aiir::Value> extents,
                  llvm::ArrayRef<aiir::Value> typeParams,
                  const std::function<decltype(genTempDeclareOp)> &genDeclare,
                  aiir::Value polymorphicMold, bool useStack = false,
                  llvm::StringRef tmpName = ".tmp.array") {
    return createAndDeclareTemp(loc, arrayType, shape, extents, typeParams,
                                genDeclare, polymorphicMold, useStack, tmpName);
  }

  /// Create an LLVM stack save intrinsic op. Returns the saved stack pointer.
  /// The stack address space is fetched from the data layout of the current
  /// module.
  aiir::Value genStackSave(aiir::Location loc);

  /// Create an LLVM stack restore intrinsic op. stackPointer should be a value
  /// previously returned from genStackSave.
  void genStackRestore(aiir::Location loc, aiir::Value stackPointer);

  /// Create a global value.
  fir::GlobalOp createGlobal(aiir::Location loc, aiir::Type type,
                             llvm::StringRef name,
                             aiir::StringAttr linkage = {},
                             aiir::Attribute value = {}, bool isConst = false,
                             bool isTarget = false,
                             cuf::DataAttributeAttr dataAttr = {});

  fir::GlobalOp createGlobal(aiir::Location loc, aiir::Type type,
                             llvm::StringRef name, bool isConst, bool isTarget,
                             std::function<void(FirOpBuilder &)> bodyBuilder,
                             aiir::StringAttr linkage = {},
                             cuf::DataAttributeAttr dataAttr = {});

  /// Create a global constant (read-only) value.
  fir::GlobalOp createGlobalConstant(aiir::Location loc, aiir::Type type,
                                     llvm::StringRef name,
                                     aiir::StringAttr linkage = {},
                                     aiir::Attribute value = {}) {
    return createGlobal(loc, type, name, linkage, value, /*isConst=*/true,
                        /*isTarget=*/false);
  }

  fir::GlobalOp
  createGlobalConstant(aiir::Location loc, aiir::Type type,
                       llvm::StringRef name,
                       std::function<void(FirOpBuilder &)> bodyBuilder,
                       aiir::StringAttr linkage = {}) {
    return createGlobal(loc, type, name, /*isConst=*/true, /*isTarget=*/false,
                        bodyBuilder, linkage);
  }

  /// Convert a StringRef string into a fir::StringLitOp.
  fir::StringLitOp createStringLitOp(aiir::Location loc,
                                     llvm::StringRef string);

  std::pair<fir::TypeInfoOp, aiir::OpBuilder::InsertPoint>
  createTypeInfoOp(aiir::Location loc, fir::RecordType recordType,
                   fir::RecordType parentType);

  //===--------------------------------------------------------------------===//
  // Linkage helpers (inline). The default linkage is external.
  //===--------------------------------------------------------------------===//

  static aiir::StringAttr createCommonLinkage(aiir::AIIRContext *context) {
    return aiir::StringAttr::get(context, "common");
  }
  aiir::StringAttr createCommonLinkage() {
    return createCommonLinkage(getContext());
  }

  aiir::StringAttr createExternalLinkage() { return getStringAttr("external"); }

  aiir::StringAttr createInternalLinkage() { return getStringAttr("internal"); }

  aiir::StringAttr createLinkOnceLinkage() { return getStringAttr("linkonce"); }

  aiir::StringAttr createLinkOnceODRLinkage() {
    return getStringAttr("linkonce_odr");
  }

  aiir::StringAttr createWeakLinkage() { return getStringAttr("weak"); }

  /// Get a function by name. If the function exists in the current module, it
  /// is returned. Otherwise, a null FuncOp is returned.
  aiir::func::FuncOp getNamedFunction(llvm::StringRef name) {
    return getNamedFunction(getModule(), getAIIRSymbolTable(), name);
  }
  static aiir::func::FuncOp
  getNamedFunction(aiir::ModuleOp module, const aiir::SymbolTable *symbolTable,
                   llvm::StringRef name);

  /// Get a function by symbol name. The result will be null if there is no
  /// function with the given symbol in the module.
  aiir::func::FuncOp getNamedFunction(aiir::SymbolRefAttr symbol) {
    return getNamedFunction(getModule(), getAIIRSymbolTable(), symbol);
  }
  static aiir::func::FuncOp
  getNamedFunction(aiir::ModuleOp module, const aiir::SymbolTable *symbolTable,
                   aiir::SymbolRefAttr symbol);

  fir::GlobalOp getNamedGlobal(llvm::StringRef name) {
    return getNamedGlobal(getModule(), getAIIRSymbolTable(), name);
  }

  static fir::GlobalOp getNamedGlobal(aiir::ModuleOp module,
                                      const aiir::SymbolTable *symbolTable,
                                      llvm::StringRef name);

  /// Lazy creation of fir.convert op.
  aiir::Value createConvert(aiir::Location loc, aiir::Type toTy,
                            aiir::Value val);

  /// Create a fir.convert op with a volatile cast if the source value's type
  /// does not match the target type's volatility.
  aiir::Value createConvertWithVolatileCast(aiir::Location loc, aiir::Type toTy,
                                            aiir::Value val);

  /// Cast \p value to have \p isVolatile volatility.
  aiir::Value createVolatileCast(aiir::Location loc, bool isVolatile,
                                 aiir::Value value);

  /// Create a fir.store of \p val into \p addr. A lazy conversion
  /// of \p val to the element type of \p addr is created if needed.
  void createStoreWithConvert(aiir::Location loc, aiir::Value val,
                              aiir::Value addr);

  /// Create a fir.load if \p val is a reference or pointer type. Return the
  /// result of the load if it was created, otherwise return \p val
  aiir::Value loadIfRef(aiir::Location loc, aiir::Value val);

  /// Determine if the named function is already in the module. Return the
  /// instance if found, otherwise add a new named function to the module.
  aiir::func::FuncOp createFunction(aiir::Location loc, llvm::StringRef name,
                                    aiir::FunctionType ty) {
    return createFunction(loc, getModule(), name, ty, getAIIRSymbolTable());
  }

  static aiir::func::FuncOp createFunction(aiir::Location loc,
                                           aiir::ModuleOp module,
                                           llvm::StringRef name,
                                           aiir::FunctionType ty,
                                           aiir::SymbolTable *);

  /// Returns a named function for a Fortran runtime API, creating
  /// it, if it does not exist in the module yet.
  /// If \p isIO is set to true, then the function corresponds
  /// to one of Fortran runtime IO APIs.
  aiir::func::FuncOp createRuntimeFunction(aiir::Location loc,
                                           llvm::StringRef name,
                                           aiir::FunctionType ty,
                                           bool isIO = false);

  /// Cast the input value to IndexType.
  aiir::Value convertToIndexType(aiir::Location loc, aiir::Value val) {
    return createConvert(loc, getIndexType(), val);
  }

  /// Construct one of the two forms of shape op from an array box.
  aiir::Value genShape(aiir::Location loc, const fir::AbstractArrayBox &arr);
  aiir::Value genShape(aiir::Location loc, llvm::ArrayRef<aiir::Value> shift,
                       llvm::ArrayRef<aiir::Value> exts);
  aiir::Value genShape(aiir::Location loc, llvm::ArrayRef<aiir::Value> exts);
  aiir::Value genShift(aiir::Location loc, llvm::ArrayRef<aiir::Value> shift);

  /// Create one of the shape ops given an extended value. For a boxed value,
  /// this may create a `fir.shift` op.
  aiir::Value createShape(aiir::Location loc, const fir::ExtendedValue &exv);

  /// Create a slice op extended value. The value to be sliced, `exv`, must be
  /// an array.
  aiir::Value createSlice(aiir::Location loc, const fir::ExtendedValue &exv,
                          aiir::ValueRange triples, aiir::ValueRange path);

  /// Create a boxed value (Fortran descriptor) to be passed to the runtime.
  /// \p exv is an extended value holding a memory reference to the object that
  /// must be boxed. This function will crash if provided something that is not
  /// a memory reference type.
  /// Array entities are boxed with a shape and possibly a shift. Character
  /// entities are boxed with a LEN parameter.
  aiir::Value createBox(aiir::Location loc, const fir::ExtendedValue &exv,
                        bool isPolymorphic = false, bool isAssumedType = false);

  aiir::Value createBox(aiir::Location loc, aiir::Type boxType,
                        aiir::Value addr, aiir::Value shape, aiir::Value slice,
                        llvm::ArrayRef<aiir::Value> lengths, aiir::Value tdesc);

  /// Create constant i1 with value 1. if \p b is true or 0. otherwise
  aiir::Value createBool(aiir::Location loc, bool b) {
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
    aiir::Operation::result_range getResults() {
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
  IfBuilder genIfOp(aiir::Location loc, aiir::TypeRange results,
                    aiir::Value cdt, bool withElseRegion) {
    auto op = fir::IfOp::create(*this, loc, results, cdt, withElseRegion);
    return IfBuilder(op, *this);
  }

  /// Create an IfOp with no "else" region, and no result values.
  /// Usage: genIfThen(loc, cdt).genThen(lambda).end();
  IfBuilder genIfThen(aiir::Location loc, aiir::Value cdt) {
    auto op = fir::IfOp::create(*this, loc, aiir::TypeRange(), cdt, false);
    return IfBuilder(op, *this);
  }

  /// Create an IfOp with an "else" region, and no result values.
  /// Usage: genIfThenElse(loc, cdt).genThen(lambda).genElse(lambda).end();
  IfBuilder genIfThenElse(aiir::Location loc, aiir::Value cdt) {
    auto op = fir::IfOp::create(*this, loc, aiir::TypeRange(), cdt, true);
    return IfBuilder(op, *this);
  }

  aiir::Value genNot(aiir::Location loc, aiir::Value boolean) {
    return aiir::arith::CmpIOp::create(*this, loc,
                                       aiir::arith::CmpIPredicate::eq, boolean,
                                       createBool(loc, false));
  }

  /// Generate code testing \p addr is not a null address.
  aiir::Value genIsNotNullAddr(aiir::Location loc, aiir::Value addr);

  /// Generate code testing \p addr is a null address.
  aiir::Value genIsNullAddr(aiir::Location loc, aiir::Value addr);

  /// Compute the extent of (lb:ub:step) as max((ub-lb+step)/step, 0). See
  /// Fortran 2018 9.5.3.3.2 section for more details.
  aiir::Value genExtentFromTriplet(aiir::Location loc, aiir::Value lb,
                                   aiir::Value ub, aiir::Value step,
                                   aiir::Type type, bool fold = false);

  /// Create an AbsentOp of \p argTy type and handle special cases, such as
  /// Character Procedure Tuple arguments.
  aiir::Value genAbsentOp(aiir::Location loc, aiir::Type argTy);

  /// Set default FastMathFlags value for all operations
  /// supporting aiir::arith::FastMathAttr that will be created
  /// by this builder.
  void setFastMathFlags(aiir::arith::FastMathFlags flags) {
    fastMathFlags = flags;
  }

  /// Set default FastMathFlags value from the passed MathOptionsBase
  /// config.
  void setFastMathFlags(Fortran::common::MathOptionsBase options);

  /// Get current FastMathFlags value.
  aiir::arith::FastMathFlags getFastMathFlags() const { return fastMathFlags; }

  /// Stringify FastMathFlags set in a way
  /// that the string may be used for mangling a function name.
  /// If FastMathFlags are set to 'none', then the result is an empty
  /// string.
  std::string getFastMathFlagsString() {
    aiir::arith::FastMathFlags flags = getFastMathFlags();
    if (flags == aiir::arith::FastMathFlags::none)
      return {};

    std::string fmfString{aiir::arith::stringifyFastMathFlags(flags)};
    std::replace(fmfString.begin(), fmfString.end(), ',', '_');
    return fmfString;
  }

  /// Set default IntegerOverflowFlags value for all operations
  /// supporting aiir::arith::IntegerOverflowFlagsAttr that will be created
  /// by this builder.
  void setIntegerOverflowFlags(aiir::arith::IntegerOverflowFlags flags) {
    integerOverflowFlags = flags;
  }

  /// Get current IntegerOverflowFlags value.
  aiir::arith::IntegerOverflowFlags getIntegerOverflowFlags() const {
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

  /// Setter/getter for fpMaxminBehavior.
  void setFPMaxminBehavior(Fortran::common::FPMaxminBehavior mode) {
    fpMaxminBehavior = mode;
  }
  Fortran::common::FPMaxminBehavior getFPMaxminBehavior() const {
    return fpMaxminBehavior;
  }

  /// Dump the current function. (debug)
  LLVM_DUMP_METHOD void dumpFunc();

  /// FirOpBuilder hook for creating new operation.
  void notifyOperationInserted(aiir::Operation *op,
                               aiir::OpBuilder::InsertPoint previous) override {
    // We only care about newly created operations.
    if (previous.isSet())
      return;
    setCommonAttributes(op);
  }

  /// Construct a data layout on demand and return it
  aiir::DataLayout &getDataLayout();

  /// Convert operands &/or result from/to unsigned so that the operation
  /// only receives/produces signless operands.
  template <typename OpTy>
  aiir::Value createUnsigned(aiir::Location loc, aiir::Type resultType,
                             aiir::Value left, aiir::Value right) {
    if (!resultType.isIntOrFloat())
      return OpTy::create(*this, loc, resultType, left, right);
    aiir::Type signlessType = aiir::IntegerType::get(
        getContext(), resultType.getIntOrFloatBitWidth(),
        aiir::IntegerType::SignednessSemantics::Signless);
    aiir::Type opResType = resultType;
    if (left.getType().isUnsignedInteger()) {
      left = createConvert(loc, signlessType, left);
      opResType = signlessType;
    }
    if (right.getType().isUnsignedInteger()) {
      right = createConvert(loc, signlessType, right);
      opResType = signlessType;
    }
    aiir::Value result = OpTy::create(*this, loc, opResType, left, right);
    if (resultType.isUnsignedInteger())
      result = createConvert(loc, resultType, result);
    return result;
  }

  /// Compare two pointer-like values using the given predicate.
  aiir::Value genPtrCompare(aiir::Location loc,
                            aiir::arith::CmpIPredicate predicate,
                            aiir::Value ptr1, aiir::Value ptr2) {
    ptr1 = createConvert(loc, getIndexType(), ptr1);
    ptr2 = createConvert(loc, getIndexType(), ptr2);
    return aiir::arith::CmpIOp::create(*this, loc, predicate, ptr1, ptr2);
  }

private:
  /// Set attributes (e.g. FastMathAttr) to \p op operation
  /// based on the current attributes setting.
  void setCommonAttributes(aiir::Operation *op) const;

  KindMapping kindMap;

  /// FastMathFlags that need to be set for operations that support
  /// aiir::arith::FastMathAttr.
  aiir::arith::FastMathFlags fastMathFlags{};

  /// Controls how max/min idioms should be implemented.
  /// Right now, it is only used to propagate FPMaxminBehavior
  /// to the IntrinsicCall lowering. In general, it can be used
  /// for generating max/min idioms through FirBuilder anywhere
  /// in the pipeline.
  Fortran::common::FPMaxminBehavior fpMaxminBehavior{
      Fortran::common::FPMaxminBehavior::Legacy};

  /// IntegerOverflowFlags that need to be set for operations that support
  /// aiir::arith::IntegerOverflowFlagsAttr.
  aiir::arith::IntegerOverflowFlags integerOverflowFlags{};

  /// Flag to control whether complex number division is lowered to a runtime
  /// function or to the AIIR complex dialect.
  bool complexDivisionToRuntimeFlag = true;

  /// fir::GlobalOp and func::FuncOp symbol table to speed-up
  /// lookups.
  aiir::SymbolTable *symbolTable = nullptr;

  /// DataLayout constructed on demand. Access via getDataLayout().
  /// Stored via a unique_ptr rather than an optional so as not to bloat this
  /// class when most instances won't ever need a data layout.
  std::unique_ptr<aiir::DataLayout> dataLayout = nullptr;
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
aiir::Value readCharLen(fir::FirOpBuilder &builder, aiir::Location loc,
                        const fir::ExtendedValue &box);

/// Read or get the extent in dimension \p dim of the array described by \p box.
aiir::Value readExtent(fir::FirOpBuilder &builder, aiir::Location loc,
                       const fir::ExtendedValue &box, unsigned dim);

/// Read or get the lower bound in dimension \p dim of the array described by
/// \p box. If the lower bound is left default in the ExtendedValue,
/// \p defaultValue will be returned.
aiir::Value readLowerBound(fir::FirOpBuilder &builder, aiir::Location loc,
                           const fir::ExtendedValue &box, unsigned dim,
                           aiir::Value defaultValue);

/// Read extents from \p box.
llvm::SmallVector<aiir::Value> readExtents(fir::FirOpBuilder &builder,
                                           aiir::Location loc,
                                           const fir::BoxValue &box);

/// Read a fir::BoxValue into an fir::UnboxValue, a fir::ArrayBoxValue or a
/// fir::CharArrayBoxValue. This should only be called if the fir::BoxValue is
/// known to be contiguous given the context (or if the resulting address will
/// not be used). If the value is polymorphic, its dynamic type will be lost.
/// This must not be used on unlimited polymorphic and assumed rank entities.
fir::ExtendedValue readBoxValue(fir::FirOpBuilder &builder, aiir::Location loc,
                                const fir::BoxValue &box);

/// Get the lower bounds of \p exv. NB: returns an empty vector if the lower
/// bounds are all ones, which is the default in Fortran.
llvm::SmallVector<aiir::Value>
getNonDefaultLowerBounds(fir::FirOpBuilder &builder, aiir::Location loc,
                         const fir::ExtendedValue &exv);

/// Return LEN parameters associated to \p exv that are not deferred (that are
/// available without having to read any fir.box values). Empty if \p exv has no
/// LEN parameters or if they are all deferred.
llvm::SmallVector<aiir::Value>
getNonDeferredLenParams(const fir::ExtendedValue &exv);

//===----------------------------------------------------------------------===//
// String literal helper helpers
//===----------------------------------------------------------------------===//

/// Create a !fir.char<1> string literal global and returns a fir::CharBoxValue
/// with its address and length.
fir::ExtendedValue createStringLiteral(fir::FirOpBuilder &, aiir::Location,
                                       llvm::StringRef string);

/// Unique a compiler generated identifier. A short prefix should be provided
/// to hint at the origin of the identifier.
std::string uniqueCGIdent(llvm::StringRef prefix, llvm::StringRef name);

/// Lowers the extents from the sequence type to Values.
/// Any unknown extents are lowered to undefined values.
llvm::SmallVector<aiir::Value> createExtents(fir::FirOpBuilder &builder,
                                             aiir::Location loc,
                                             fir::SequenceType seqTy);

//===--------------------------------------------------------------------===//
// Location helpers
//===--------------------------------------------------------------------===//

/// Generate a string literal containing the file name and return its address
aiir::Value locationToFilename(fir::FirOpBuilder &, aiir::Location);
/// Generate a constant of the given type with the location line number
aiir::Value locationToLineNo(fir::FirOpBuilder &, aiir::Location, aiir::Type);

//===--------------------------------------------------------------------===//
// ExtendedValue helpers
//===--------------------------------------------------------------------===//

/// Return the extended value for a component of a derived type instance given
/// the address of the component.
fir::ExtendedValue componentToExtendedValue(fir::FirOpBuilder &builder,
                                            aiir::Location loc,
                                            aiir::Value component);

/// Given the address of an array element and the ExtendedValue describing the
/// array, returns the ExtendedValue describing the array element. The purpose
/// is to propagate the LEN parameters of the array to the element. This can be
/// used for elements of `array` or `array(i:j:k)`. If \p element belongs to an
/// array section `array%x` whose base is \p array,
/// arraySectionElementToExtendedValue must be used instead.
fir::ExtendedValue arrayElementToExtendedValue(fir::FirOpBuilder &builder,
                                               aiir::Location loc,
                                               const fir::ExtendedValue &array,
                                               aiir::Value element);

/// Build the ExtendedValue for \p element that is an element of an array or
/// array section with \p array base (`array` or `array(i:j:k)%x%y`).
/// If it is an array section, \p slice must be provided and be a fir::SliceOp
/// that describes the section.
fir::ExtendedValue arraySectionElementToExtendedValue(
    fir::FirOpBuilder &builder, aiir::Location loc,
    const fir::ExtendedValue &array, aiir::Value element, aiir::Value slice);

/// Assign \p rhs to \p lhs. Both \p rhs and \p lhs must be scalars. The
/// assignment follows Fortran intrinsic assignment semantic (10.2.1.3).
void genScalarAssignment(fir::FirOpBuilder &builder, aiir::Location loc,
                         const fir::ExtendedValue &lhs,
                         const fir::ExtendedValue &rhs,
                         bool needFinalization = false,
                         bool isTemporaryLHS = false,
                         aiir::ArrayAttr accessGroups = {});

/// Assign \p rhs to \p lhs. Both \p rhs and \p lhs must be scalar derived
/// types. The assignment follows Fortran intrinsic assignment semantic for
/// derived types (10.2.1.3 point 13).
void genRecordAssignment(fir::FirOpBuilder &builder, aiir::Location loc,
                         const fir::ExtendedValue &lhs,
                         const fir::ExtendedValue &rhs,
                         bool needFinalization = false,
                         bool isTemporaryLHS = false);

/// Builds and returns the type of a ragged array header used to cache mask
/// evaluations. RaggedArrayHeader is defined in
/// flang/include/flang/Runtime/ragged.h.
aiir::TupleType getRaggedArrayHeaderType(fir::FirOpBuilder &builder);

/// Generate the, possibly dynamic, LEN of a CHARACTER. \p arrLoad determines
/// the base array. After applying \p path, the result must be a reference to a
/// `!fir.char` type object. \p substring must have 0, 1, or 2 members. The
/// first member is the starting offset. The second is the ending offset.
aiir::Value genLenOfCharacter(fir::FirOpBuilder &builder, aiir::Location loc,
                              fir::ArrayLoadOp arrLoad,
                              llvm::ArrayRef<aiir::Value> path,
                              llvm::ArrayRef<aiir::Value> substring);
aiir::Value genLenOfCharacter(fir::FirOpBuilder &builder, aiir::Location loc,
                              fir::SequenceType seqTy, aiir::Value memref,
                              llvm::ArrayRef<aiir::Value> typeParams,
                              llvm::ArrayRef<aiir::Value> path,
                              llvm::ArrayRef<aiir::Value> substring);

/// Create the zero value of a given the numerical or logical \p type (`false`
/// for logical types).
aiir::Value createZeroValue(fir::FirOpBuilder &builder, aiir::Location loc,
                            aiir::Type type);

/// Create a one value of a given numerical or logical \p type (`true`
/// for logical types).
aiir::Value createOneValue(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::Type type);

/// Get the integer constants of triplet and compute the extent.
std::optional<std::int64_t> getExtentFromTriplet(aiir::Value lb, aiir::Value ub,
                                                 aiir::Value stride);

/// Compute the extent value given the lower bound \lb and upper bound \ub.
/// All inputs must have the same SSA integer type.
aiir::Value computeExtent(fir::FirOpBuilder &builder, aiir::Location loc,
                          aiir::Value lb, aiir::Value ub);
aiir::Value computeExtent(fir::FirOpBuilder &builder, aiir::Location loc,
                          aiir::Value lb, aiir::Value ub, aiir::Value zero,
                          aiir::Value one);

/// Generate max(\p value, 0) where \p value is a scalar integer.
aiir::Value genMaxWithZero(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::Value value);
aiir::Value genMaxWithZero(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::Value value, aiir::Value zero);

/// The type(C_PTR/C_FUNPTR) is defined as the derived type with only one
/// component of integer 64, and the component is the C address. Get the C
/// address.
aiir::Value genCPtrOrCFunptrAddr(fir::FirOpBuilder &builder, aiir::Location loc,
                                 aiir::Value cPtr, aiir::Type ty);

/// The type(C_DEVPTR) is defined as the derived type with only one
/// component of C_PTR type. Get the C address from the C_PTR component.
aiir::Value genCDevPtrAddr(fir::FirOpBuilder &builder, aiir::Location loc,
                           aiir::Value cDevPtr, aiir::Type ty);

/// Get the C address value.
aiir::Value genCPtrOrCFunptrValue(fir::FirOpBuilder &builder,
                                  aiir::Location loc, aiir::Value cPtr);

/// Create a fir.box from a fir::ExtendedValue and wrap it in a fir::BoxValue
/// to keep all the lower bound and explicit parameter information.
fir::BoxValue createBoxValue(fir::FirOpBuilder &builder, aiir::Location loc,
                             const fir::ExtendedValue &exv);

/// Generate Null BoxProc for procedure pointer null initialization.
aiir::Value createNullBoxProc(fir::FirOpBuilder &builder, aiir::Location loc,
                              aiir::Type boxType);

/// Convert a value to a new type. Return the value directly if it has the right
/// type.
aiir::Value createConvert(aiir::OpBuilder &, aiir::Location, aiir::Type,
                          aiir::Value);

/// Set internal linkage attribute on a function.
void setInternalLinkage(aiir::func::FuncOp);

llvm::SmallVector<aiir::Value>
elideExtentsAlreadyInType(aiir::Type type, aiir::ValueRange shape);

llvm::SmallVector<aiir::Value>
elideLengthsAlreadyInType(aiir::Type type, aiir::ValueRange lenParams);

/// Get the address space which should be used for allocas
uint64_t getAllocaAddressSpace(const aiir::DataLayout *dataLayout);

/// The two vectors of AIIR values have the following property:
///   \p extents1[i] must have the same value as \p extents2[i]
/// The function returns a new vector of AIIR values that preserves
/// the same property vs \p extents1 and \p extents2, but allows
/// more optimizations. For example, if extents1[j] is a known constant,
/// and extents2[j] is not, then result[j] is the AIIR value extents1[j].
llvm::SmallVector<aiir::Value> deduceOptimalExtents(aiir::ValueRange extents1,
                                                    aiir::ValueRange extents2);

uint64_t getGlobalAddressSpace(aiir::DataLayout *dataLayout);

uint64_t getProgramAddressSpace(aiir::DataLayout *dataLayout);

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
llvm::SmallVector<aiir::Value> updateRuntimeExtentsForEmptyArrays(
    fir::FirOpBuilder &builder, aiir::Location loc, aiir::ValueRange extents);

/// Given \p box of type fir::BaseBoxType representing an array,
/// the function generates code to fetch the lower bounds,
/// the extents and the strides from the box. The values are returned via
/// \p lbounds, \p extents and \p strides.
void genDimInfoFromBox(fir::FirOpBuilder &builder, aiir::Location loc,
                       aiir::Value box,
                       llvm::SmallVectorImpl<aiir::Value> *lbounds,
                       llvm::SmallVectorImpl<aiir::Value> *extents,
                       llvm::SmallVectorImpl<aiir::Value> *strides);

/// Generate an LLVM dialect lifetime start marker at the current insertion
/// point given an fir.alloca. Returns the value to be passed to the lifetime
/// end marker.
aiir::Value genLifetimeStart(aiir::OpBuilder &builder, aiir::Location loc,
                             fir::AllocaOp alloc, const aiir::DataLayout *dl);

/// Generate an LLVM dialect lifetime end marker at the current insertion point
/// given an llvm.ptr value.
void genLifetimeEnd(aiir::OpBuilder &builder, aiir::Location loc,
                    aiir::Value mem);

/// Given a fir.box or fir.class \p box describing an entity and a raw address
/// \p newAddr for an entity with the same Fortran properties (rank, dynamic
/// type, length parameters and bounds) and attributes (POINTER or ALLOCATABLE),
/// create a box for \p newAddr with the same type as \p box. This assumes \p
/// newAddr is for contiguous storage (\p box does not have to be contiguous).
aiir::Value getDescriptorWithNewBaseAddress(fir::FirOpBuilder &builder,
                                            aiir::Location loc, aiir::Value box,
                                            aiir::Value newAddr);

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_FIRBUILDER_H
