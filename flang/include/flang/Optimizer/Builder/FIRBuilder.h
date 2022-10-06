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
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

namespace fir {
class AbstractArrayBox;
class ExtendedValue;
class BoxValue;

//===----------------------------------------------------------------------===//
// FirOpBuilder
//===----------------------------------------------------------------------===//

/// Extends the MLIR OpBuilder to provide methods for building common FIR
/// patterns.
class FirOpBuilder : public mlir::OpBuilder {
public:
  explicit FirOpBuilder(mlir::Operation *op, const fir::KindMapping &kindMap)
      : OpBuilder{op}, kindMap{kindMap} {}
  explicit FirOpBuilder(mlir::OpBuilder &builder,
                        const fir::KindMapping &kindMap)
      : OpBuilder{builder}, kindMap{kindMap} {}

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
                                   bool allowCharacterConversion = false);

  /// Get the entry block of the current Function
  mlir::Block *getEntryBlock() { return &getFunction().front(); }

  /// Get the block for adding Allocas. If OpenMP is enabled then get the
  /// the alloca block from an Operation which can be Outlined. Otherwise
  /// use the entry block of the current Function
  mlir::Block *getAllocaBlock();

  /// Safely create a reference type to the type `eleTy`.
  mlir::Type getRefType(mlir::Type eleTy);

  /// Create a sequence of `eleTy` with `rank` dimensions of unknown size.
  mlir::Type getVarLenSeqTy(mlir::Type eleTy, unsigned rank = 1);

  /// Get character length type.
  mlir::Type getCharacterLengthType() { return getIndexType(); }

  /// Get the integer type whose bit width corresponds to the width of pointer
  /// types, or is bigger.
  mlir::Type getIntPtrType() {
    // TODO: Delay the need of such type until codegen or find a way to use
    // llvm::DataLayout::getPointerSizeInBits here.
    return getI64Type();
  }

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
  mlir::Value createIntegerConstant(mlir::Location loc, mlir::Type integerType,
                                    std::int64_t i);

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

  /// Create a temporary. A temp is allocated using `fir.alloca` and can be read
  /// and written using `fir.load` and `fir.store`, resp.  The temporary can be
  /// given a name via a front-end `Symbol` or a `StringRef`.
  mlir::Value createTemporary(mlir::Location loc, mlir::Type type,
                              llvm::StringRef name = {},
                              mlir::ValueRange shape = {},
                              mlir::ValueRange lenParams = {},
                              llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

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

  /// Create a global value.
  fir::GlobalOp createGlobal(mlir::Location loc, mlir::Type type,
                             llvm::StringRef name,
                             mlir::StringAttr linkage = {},
                             mlir::Attribute value = {}, bool isConst = false);

  fir::GlobalOp createGlobal(mlir::Location loc, mlir::Type type,
                             llvm::StringRef name, bool isConst,
                             std::function<void(FirOpBuilder &)> bodyBuilder,
                             mlir::StringAttr linkage = {});

  /// Create a global constant (read-only) value.
  fir::GlobalOp createGlobalConstant(mlir::Location loc, mlir::Type type,
                                     llvm::StringRef name,
                                     mlir::StringAttr linkage = {},
                                     mlir::Attribute value = {}) {
    return createGlobal(loc, type, name, linkage, value, /*isConst=*/true);
  }

  fir::GlobalOp
  createGlobalConstant(mlir::Location loc, mlir::Type type,
                       llvm::StringRef name,
                       std::function<void(FirOpBuilder &)> bodyBuilder,
                       mlir::StringAttr linkage = {}) {
    return createGlobal(loc, type, name, /*isConst=*/true, bodyBuilder,
                        linkage);
  }

  /// Convert a StringRef string into a fir::StringLitOp.
  fir::StringLitOp createStringLitOp(mlir::Location loc,
                                     llvm::StringRef string);

  //===--------------------------------------------------------------------===//
  // Linkage helpers (inline). The default linkage is external.
  //===--------------------------------------------------------------------===//

  mlir::StringAttr createCommonLinkage() { return getStringAttr("common"); }

  mlir::StringAttr createInternalLinkage() { return getStringAttr("internal"); }

  mlir::StringAttr createLinkOnceLinkage() { return getStringAttr("linkonce"); }

  mlir::StringAttr createLinkOnceODRLinkage() {
    return getStringAttr("linkonce_odr");
  }

  mlir::StringAttr createWeakLinkage() { return getStringAttr("weak"); }

  /// Get a function by name. If the function exists in the current module, it
  /// is returned. Otherwise, a null FuncOp is returned.
  mlir::func::FuncOp getNamedFunction(llvm::StringRef name) {
    return getNamedFunction(getModule(), name);
  }
  static mlir::func::FuncOp getNamedFunction(mlir::ModuleOp module,
                                             llvm::StringRef name);

  /// Get a function by symbol name. The result will be null if there is no
  /// function with the given symbol in the module.
  mlir::func::FuncOp getNamedFunction(mlir::SymbolRefAttr symbol) {
    return getNamedFunction(getModule(), symbol);
  }
  static mlir::func::FuncOp getNamedFunction(mlir::ModuleOp module,
                                             mlir::SymbolRefAttr symbol);

  fir::GlobalOp getNamedGlobal(llvm::StringRef name) {
    return getNamedGlobal(getModule(), name);
  }

  static fir::GlobalOp getNamedGlobal(mlir::ModuleOp module,
                                      llvm::StringRef name);

  /// Lazy creation of fir.convert op.
  mlir::Value createConvert(mlir::Location loc, mlir::Type toTy,
                            mlir::Value val);

  /// Create a fir.store of \p val into \p addr. A lazy conversion
  /// of \p val to the element type of \p addr is created if needed.
  void createStoreWithConvert(mlir::Location loc, mlir::Value val,
                              mlir::Value addr);

  /// Create a new FuncOp. If the function may have already been created, use
  /// `addNamedFunction` instead.
  mlir::func::FuncOp createFunction(mlir::Location loc, llvm::StringRef name,
                                    mlir::FunctionType ty) {
    return createFunction(loc, getModule(), name, ty);
  }

  static mlir::func::FuncOp createFunction(mlir::Location loc,
                                           mlir::ModuleOp module,
                                           llvm::StringRef name,
                                           mlir::FunctionType ty);

  /// Determine if the named function is already in the module. Return the
  /// instance if found, otherwise add a new named function to the module.
  mlir::func::FuncOp addNamedFunction(mlir::Location loc, llvm::StringRef name,
                                      mlir::FunctionType ty) {
    if (auto func = getNamedFunction(name))
      return func;
    return createFunction(loc, name, ty);
  }

  static mlir::func::FuncOp addNamedFunction(mlir::Location loc,
                                             mlir::ModuleOp module,
                                             llvm::StringRef name,
                                             mlir::FunctionType ty) {
    if (auto func = getNamedFunction(module, name))
      return func;
    return createFunction(loc, module, name, ty);
  }

  /// Cast the input value to IndexType.
  mlir::Value convertToIndexType(mlir::Location loc, mlir::Value val) {
    return createConvert(loc, getIndexType(), val);
  }

  /// Construct one of the two forms of shape op from an array box.
  mlir::Value genShape(mlir::Location loc, const fir::AbstractArrayBox &arr);
  mlir::Value genShape(mlir::Location loc, llvm::ArrayRef<mlir::Value> shift,
                       llvm::ArrayRef<mlir::Value> exts);
  mlir::Value genShape(mlir::Location loc, llvm::ArrayRef<mlir::Value> exts);

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
  mlir::Value createBox(mlir::Location loc, const fir::ExtendedValue &exv);

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
    auto op = create<fir::IfOp>(loc, results, cdt, withElseRegion);
    return IfBuilder(op, *this);
  }

  /// Create an IfOp with no "else" region, and no result values.
  /// Usage: genIfThen(loc, cdt).genThen(lambda).end();
  IfBuilder genIfThen(mlir::Location loc, mlir::Value cdt) {
    auto op = create<fir::IfOp>(loc, llvm::None, cdt, false);
    return IfBuilder(op, *this);
  }

  /// Create an IfOp with an "else" region, and no result values.
  /// Usage: genIfThenElse(loc, cdt).genThen(lambda).genElse(lambda).end();
  IfBuilder genIfThenElse(mlir::Location loc, mlir::Value cdt) {
    auto op = create<fir::IfOp>(loc, llvm::None, cdt, true);
    return IfBuilder(op, *this);
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

  /// Dump the current function. (debug)
  LLVM_DUMP_METHOD void dumpFunc();

private:
  const KindMapping &kindMap;
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
                         const fir::ExtendedValue &rhs);
/// Assign \p rhs to \p lhs. Both \p rhs and \p lhs must be scalar derived
/// types. The assignment follows Fortran intrinsic assignment semantic for
/// derived types (10.2.1.3 point 13).
void genRecordAssignment(fir::FirOpBuilder &builder, mlir::Location loc,
                         const fir::ExtendedValue &lhs,
                         const fir::ExtendedValue &rhs);

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

/// Unwrap integer constant from an mlir::Value.
llvm::Optional<std::int64_t> getIntIfConstant(mlir::Value value);

/// Get the integer constants of triplet and compute the extent.
llvm::Optional<std::int64_t>
getExtentFromTriplet(mlir::Value lb, mlir::Value ub, mlir::Value stride);

/// Generate max(\p value, 0) where \p value is a scalar integer.
mlir::Value genMaxWithZero(fir::FirOpBuilder &builder, mlir::Location loc,
                           mlir::Value value);

/// The type(C_PTR/C_FUNPTR) is defined as the derived type with only one
/// component of integer 64, and the component is the C address. Get the C
/// address.
mlir::Value genCPtrOrCFunptrAddr(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Value cPtr, mlir::Type ty);

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_FIRBUILDER_H
