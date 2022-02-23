//===-- ConvertExpr.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/ConvertExpr.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/CallInterface.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/ConvertVariable.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-expr"

//===----------------------------------------------------------------------===//
// The composition and structure of Fortran::evaluate::Expr is defined in
// the various header files in include/flang/Evaluate. You are referred
// there for more information on these data structures. Generally speaking,
// these data structures are a strongly typed family of abstract data types
// that, composed as trees, describe the syntax of Fortran expressions.
//
// This part of the bridge can traverse these tree structures and lower them
// to the correct FIR representation in SSA form.
//===----------------------------------------------------------------------===//

/// Place \p exv in memory if it is not already a memory reference. If
/// \p forceValueType is provided, the value is first casted to the provided
/// type before being stored (this is mainly intended for logicals whose value
/// may be `i1` but needed to be stored as Fortran logicals).
static fir::ExtendedValue
placeScalarValueInMemory(fir::FirOpBuilder &builder, mlir::Location loc,
                         const fir::ExtendedValue &exv,
                         mlir::Type storageType) {
  mlir::Value valBase = fir::getBase(exv);
  if (fir::conformsWithPassByRef(valBase.getType()))
    return exv;

  assert(!fir::hasDynamicSize(storageType) &&
         "only expect statically sized scalars to be by value");

  // Since `a` is not itself a valid referent, determine its value and
  // create a temporary location at the beginning of the function for
  // referencing.
  mlir::Value val = builder.createConvert(loc, storageType, valBase);
  mlir::Value temp = builder.createTemporary(
      loc, storageType,
      llvm::ArrayRef<mlir::NamedAttribute>{
          Fortran::lower::getAdaptToByRefAttr(builder)});
  builder.create<fir::StoreOp>(loc, val, temp);
  return fir::substBase(exv, temp);
}

/// Is this a variable wrapped in parentheses?
template <typename A>
static bool isParenthesizedVariable(const A &) {
  return false;
}
template <typename T>
static bool isParenthesizedVariable(const Fortran::evaluate::Expr<T> &expr) {
  using ExprVariant = decltype(Fortran::evaluate::Expr<T>::u);
  using Parentheses = Fortran::evaluate::Parentheses<T>;
  if constexpr (Fortran::common::HasMember<Parentheses, ExprVariant>) {
    if (const auto *parentheses = std::get_if<Parentheses>(&expr.u))
      return Fortran::evaluate::IsVariable(parentheses->left());
    return false;
  } else {
    return std::visit([&](const auto &x) { return isParenthesizedVariable(x); },
                      expr.u);
  }
}

/// Generate a load of a value from an address. Beware that this will lose
/// any dynamic type information for polymorphic entities (note that unlimited
/// polymorphic cannot be loaded and must not be provided here).
static fir::ExtendedValue genLoad(fir::FirOpBuilder &builder,
                                  mlir::Location loc,
                                  const fir::ExtendedValue &addr) {
  return addr.match(
      [](const fir::CharBoxValue &box) -> fir::ExtendedValue { return box; },
      [&](const fir::UnboxedValue &v) -> fir::ExtendedValue {
        if (fir::unwrapRefType(fir::getBase(v).getType())
                .isa<fir::RecordType>())
          return v;
        return builder.create<fir::LoadOp>(loc, fir::getBase(v));
      },
      [&](const fir::MutableBoxValue &box) -> fir::ExtendedValue {
        TODO(loc, "genLoad for MutableBoxValue");
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        TODO(loc, "genLoad for BoxValue");
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(
            loc, "attempting to load whole array or procedure address");
      });
}

/// Is this a call to an elemental procedure with at least one array argument?
static bool
isElementalProcWithArrayArgs(const Fortran::evaluate::ProcedureRef &procRef) {
  if (procRef.IsElemental())
    for (const std::optional<Fortran::evaluate::ActualArgument> &arg :
         procRef.arguments())
      if (arg && arg->Rank() != 0)
        return true;
  return false;
}

/// If \p arg is the address of a function with a denoted host-association tuple
/// argument, then return the host-associations tuple value of the current
/// procedure. Otherwise, return nullptr.
static mlir::Value
argumentHostAssocs(Fortran::lower::AbstractConverter &converter,
                   mlir::Value arg) {
  if (auto addr = mlir::dyn_cast_or_null<fir::AddrOfOp>(arg.getDefiningOp())) {
    auto &builder = converter.getFirOpBuilder();
    if (auto funcOp = builder.getNamedFunction(addr.getSymbol()))
      if (fir::anyFuncArgsHaveAttr(funcOp, fir::getHostAssocAttrName()))
        return converter.hostAssocTupleValue();
  }
  return {};
}

namespace {

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ScalarExprLowering {
public:
  using ExtValue = fir::ExtendedValue;

  explicit ScalarExprLowering(mlir::Location loc,
                              Fortran::lower::AbstractConverter &converter,
                              Fortran::lower::SymMap &symMap,
                              Fortran::lower::StatementContext &stmtCtx)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, stmtCtx{stmtCtx}, symMap{symMap} {
  }

  ExtValue genExtAddr(const Fortran::lower::SomeExpr &expr) {
    return gen(expr);
  }

  /// Lower `expr` to be passed as a fir.box argument. Do not create a temp
  /// for the expr if it is a variable that can be described as a fir.box.
  ExtValue genBoxArg(const Fortran::lower::SomeExpr &expr) {
    bool saveUseBoxArg = useBoxArg;
    useBoxArg = true;
    ExtValue result = gen(expr);
    useBoxArg = saveUseBoxArg;
    return result;
  }

  ExtValue genExtValue(const Fortran::lower::SomeExpr &expr) {
    return genval(expr);
  }

  mlir::Location getLoc() { return location; }

  template <typename A>
  mlir::Value genunbox(const A &expr) {
    ExtValue e = genval(expr);
    if (const fir::UnboxedValue *r = e.getUnboxed())
      return *r;
    fir::emitFatalError(getLoc(), "unboxed expression expected");
  }

  /// Generate an integral constant of `value`
  template <int KIND>
  mlir::Value genIntegerConstant(mlir::MLIRContext *context,
                                 std::int64_t value) {
    mlir::Type type =
        converter.genType(Fortran::common::TypeCategory::Integer, KIND);
    return builder.createIntegerConstant(getLoc(), type, value);
  }

  /// Generate a logical/boolean constant of `value`
  mlir::Value genBoolConstant(bool value) {
    return builder.createBool(getLoc(), value);
  }

  /// Generate a real constant with a value `value`.
  template <int KIND>
  mlir::Value genRealConstant(mlir::MLIRContext *context,
                              const llvm::APFloat &value) {
    mlir::Type fltTy = Fortran::lower::convertReal(context, KIND);
    return builder.createRealConstant(getLoc(), fltTy, value);
  }

  /// Returns a reference to a symbol or its box/boxChar descriptor if it has
  /// one.
  ExtValue gen(Fortran::semantics::SymbolRef sym) {
    if (Fortran::lower::SymbolBox val = symMap.lookupSymbol(sym))
      return val.match([&val](auto &) { return val.toExtendedValue(); });
    LLVM_DEBUG(llvm::dbgs()
               << "unknown symbol: " << sym << "\nmap: " << symMap << '\n');
    fir::emitFatalError(getLoc(), "symbol is not mapped to any IR value");
  }

  ExtValue genLoad(const ExtValue &exv) {
    return ::genLoad(builder, getLoc(), exv);
  }

  ExtValue genval(Fortran::semantics::SymbolRef sym) {
    ExtValue var = gen(sym);
    if (const fir::UnboxedValue *s = var.getUnboxed())
      if (fir::isReferenceLike(s->getType()))
        return genLoad(*s);
    return var;
  }

  ExtValue genval(const Fortran::evaluate::BOZLiteralConstant &) {
    TODO(getLoc(), "genval BOZ");
  }

  /// Return indirection to function designated in ProcedureDesignator.
  /// The type of the function indirection is not guaranteed to match the one
  /// of the ProcedureDesignator due to Fortran implicit typing rules.
  ExtValue genval(const Fortran::evaluate::ProcedureDesignator &proc) {
    TODO(getLoc(), "genval ProcedureDesignator");
  }

  ExtValue genval(const Fortran::evaluate::NullPointer &) {
    TODO(getLoc(), "genval NullPointer");
  }

  ExtValue genval(const Fortran::evaluate::StructureConstructor &ctor) {
    TODO(getLoc(), "genval StructureConstructor");
  }

  /// Lowering of an <i>ac-do-variable</i>, which is not a Symbol.
  ExtValue genval(const Fortran::evaluate::ImpliedDoIndex &var) {
    TODO(getLoc(), "genval ImpliedDoIndex");
  }

  ExtValue genval(const Fortran::evaluate::DescriptorInquiry &desc) {
    TODO(getLoc(), "genval DescriptorInquiry");
  }

  ExtValue genval(const Fortran::evaluate::TypeParamInquiry &) {
    TODO(getLoc(), "genval TypeParamInquiry");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexComponent<KIND> &part) {
    TODO(getLoc(), "genval ComplexComponent");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    mlir::Value input = genunbox(op.left());
    // Like LLVM, integer negation is the binary op "0 - value"
    mlir::Value zero = genIntegerConstant<KIND>(builder.getContext(), 0);
    return builder.create<mlir::arith::SubIOp>(getLoc(), zero, input);
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    return builder.create<mlir::arith::NegFOp>(getLoc(), genunbox(op.left()));
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Negate<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Complex, KIND>> &op) {
    return builder.create<fir::NegcOp>(getLoc(), genunbox(op.left()));
  }

  template <typename OpTy>
  mlir::Value createBinaryOp(const ExtValue &left, const ExtValue &right) {
    assert(fir::isUnboxedValue(left) && fir::isUnboxedValue(right));
    mlir::Value lhs = fir::getBase(left);
    mlir::Value rhs = fir::getBase(right);
    assert(lhs.getType() == rhs.getType() && "types must be the same");
    return builder.create<OpTy>(getLoc(), lhs, rhs);
  }

  template <typename OpTy, typename A>
  mlir::Value createBinaryOp(const A &ex) {
    ExtValue left = genval(ex.left());
    return createBinaryOp<OpTy>(left, genval(ex.right()));
  }

#undef GENBIN
#define GENBIN(GenBinEvOp, GenBinTyCat, GenBinFirOp)                           \
  template <int KIND>                                                          \
  ExtValue genval(const Fortran::evaluate::GenBinEvOp<Fortran::evaluate::Type< \
                      Fortran::common::TypeCategory::GenBinTyCat, KIND>> &x) { \
    return createBinaryOp<GenBinFirOp>(x);                                     \
  }

  GENBIN(Add, Integer, mlir::arith::AddIOp)
  GENBIN(Add, Real, mlir::arith::AddFOp)
  GENBIN(Add, Complex, fir::AddcOp)
  GENBIN(Subtract, Integer, mlir::arith::SubIOp)
  GENBIN(Subtract, Real, mlir::arith::SubFOp)
  GENBIN(Subtract, Complex, fir::SubcOp)
  GENBIN(Multiply, Integer, mlir::arith::MulIOp)
  GENBIN(Multiply, Real, mlir::arith::MulFOp)
  GENBIN(Multiply, Complex, fir::MulcOp)
  GENBIN(Divide, Integer, mlir::arith::DivSIOp)
  GENBIN(Divide, Real, mlir::arith::DivFOp)
  GENBIN(Divide, Complex, fir::DivcOp)

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::Power<Fortran::evaluate::Type<TC, KIND>> &op) {
    TODO(getLoc(), "genval Power");
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genval(
      const Fortran::evaluate::RealToIntPower<Fortran::evaluate::Type<TC, KIND>>
          &op) {
    TODO(getLoc(), "genval RealToInt");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::ComplexConstructor<KIND> &op) {
    mlir::Value realPartValue = genunbox(op.left());
    return fir::factory::Complex{builder, getLoc()}.createComplex(
        KIND, realPartValue, genunbox(op.right()));
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Concat<KIND> &op) {
    TODO(getLoc(), "genval Concat<KIND>");
  }

  /// MIN and MAX operations
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Extremum<Fortran::evaluate::Type<TC, KIND>>
             &op) {
    TODO(getLoc(), "genval Extremum<TC, KIND>");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::SetLength<KIND> &x) {
    TODO(getLoc(), "genval SetLength<KIND>");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Integer, KIND>> &op) {
    TODO(getLoc(), "genval integer comparison");
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Real, KIND>> &op) {
    TODO(getLoc(), "genval real comparison");
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Complex, KIND>> &op) {
    TODO(getLoc(), "genval complex comparison");
  }
  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Relational<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Character, KIND>> &op) {
    TODO(getLoc(), "genval char comparison");
  }

  ExtValue
  genval(const Fortran::evaluate::Relational<Fortran::evaluate::SomeType> &op) {
    TODO(getLoc(), "genval comparison");
  }

  template <Fortran::common::TypeCategory TC1, int KIND,
            Fortran::common::TypeCategory TC2>
  ExtValue
  genval(const Fortran::evaluate::Convert<Fortran::evaluate::Type<TC1, KIND>,
                                          TC2> &convert) {
    mlir::Type ty = converter.genType(TC1, KIND);
    mlir::Value operand = genunbox(convert.left());
    return builder.convertWithSemantics(getLoc(), ty, operand);
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Parentheses<A> &op) {
    TODO(getLoc(), "genval parentheses<A>");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Not<KIND> &op) {
    TODO(getLoc(), "genval Not<KIND>");
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::LogicalOperation<KIND> &op) {
    TODO(getLoc(), "genval LogicalOperation<KIND>");
  }

  /// Convert a scalar literal constant to IR.
  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue genScalarLit(
      const Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>
          &value) {
    if constexpr (TC == Fortran::common::TypeCategory::Integer) {
      return genIntegerConstant<KIND>(builder.getContext(), value.ToInt64());
    } else if constexpr (TC == Fortran::common::TypeCategory::Logical) {
      return genBoolConstant(value.IsTrue());
    } else if constexpr (TC == Fortran::common::TypeCategory::Real) {
      std::string str = value.DumpHexadecimal();
      if constexpr (KIND == 2) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEhalf(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 3) {
        llvm::APFloat floatVal{llvm::APFloatBase::BFloat(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 4) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEsingle(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 10) {
        llvm::APFloat floatVal{llvm::APFloatBase::x87DoubleExtended(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else if constexpr (KIND == 16) {
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEquad(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      } else {
        // convert everything else to double
        llvm::APFloat floatVal{llvm::APFloatBase::IEEEdouble(), str};
        return genRealConstant<KIND>(builder.getContext(), floatVal);
      }
    } else if constexpr (TC == Fortran::common::TypeCategory::Complex) {
      using TR =
          Fortran::evaluate::Type<Fortran::common::TypeCategory::Real, KIND>;
      Fortran::evaluate::ComplexConstructor<KIND> ctor(
          Fortran::evaluate::Expr<TR>{
              Fortran::evaluate::Constant<TR>{value.REAL()}},
          Fortran::evaluate::Expr<TR>{
              Fortran::evaluate::Constant<TR>{value.AIMAG()}});
      return genunbox(ctor);
    } else /*constexpr*/ {
      llvm_unreachable("unhandled constant");
    }
  }

  /// Convert a ascii scalar literal CHARACTER to IR. (specialization)
  ExtValue
  genAsciiScalarLit(const Fortran::evaluate::Scalar<Fortran::evaluate::Type<
                        Fortran::common::TypeCategory::Character, 1>> &value,
                    int64_t len) {
    assert(value.size() == static_cast<std::uint64_t>(len) &&
           "value.size() doesn't match with len");
    return fir::factory::createStringLiteral(builder, getLoc(), value);
  }

  template <Fortran::common::TypeCategory TC, int KIND>
  ExtValue
  genval(const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>>
             &con) {
    if (con.Rank() > 0)
      TODO(getLoc(), "genval array constant");
    std::optional<Fortran::evaluate::Scalar<Fortran::evaluate::Type<TC, KIND>>>
        opt = con.GetScalarValue();
    assert(opt.has_value() && "constant has no value");
    if constexpr (TC == Fortran::common::TypeCategory::Character) {
      if constexpr (KIND == 1)
        return genAsciiScalarLit(opt.value(), con.LEN());
      TODO(getLoc(), "genval for Character with KIND != 1");
    } else {
      return genScalarLit<TC, KIND>(opt.value());
    }
  }

  fir::ExtendedValue genval(
      const Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived> &con) {
    TODO(getLoc(), "genval constant derived");
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::ArrayConstructor<A> &) {
    TODO(getLoc(), "genval ArrayConstructor<A>");
  }

  ExtValue gen(const Fortran::evaluate::ComplexPart &x) {
    TODO(getLoc(), "gen ComplexPart");
  }
  ExtValue genval(const Fortran::evaluate::ComplexPart &x) {
    TODO(getLoc(), "genval ComplexPart");
  }

  ExtValue gen(const Fortran::evaluate::Substring &s) {
    TODO(getLoc(), "gen Substring");
  }
  ExtValue genval(const Fortran::evaluate::Substring &ss) {
    TODO(getLoc(), "genval Substring");
  }

  ExtValue genval(const Fortran::evaluate::Subscript &subs) {
    TODO(getLoc(), "genval Subscript");
  }

  ExtValue gen(const Fortran::evaluate::DataRef &dref) {
    TODO(getLoc(), "gen DataRef");
  }
  ExtValue genval(const Fortran::evaluate::DataRef &dref) {
    TODO(getLoc(), "genval DataRef");
  }

  ExtValue gen(const Fortran::evaluate::Component &cmpt) {
    TODO(getLoc(), "gen Component");
  }
  ExtValue genval(const Fortran::evaluate::Component &cmpt) {
    TODO(getLoc(), "genval Component");
  }

  ExtValue genval(const Fortran::semantics::Bound &bound) {
    TODO(getLoc(), "genval Bound");
  }

  ExtValue gen(const Fortran::evaluate::ArrayRef &aref) {
    TODO(getLoc(), "gen ArrayRef");
  }
  ExtValue genval(const Fortran::evaluate::ArrayRef &aref) {
    TODO(getLoc(), "genval ArrayRef");
  }

  ExtValue gen(const Fortran::evaluate::CoarrayRef &coref) {
    TODO(getLoc(), "gen CoarrayRef");
  }
  ExtValue genval(const Fortran::evaluate::CoarrayRef &coref) {
    TODO(getLoc(), "genval CoarrayRef");
  }

  template <typename A>
  ExtValue gen(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return gen(x); }, des.u);
  }
  template <typename A>
  ExtValue genval(const Fortran::evaluate::Designator<A> &des) {
    return std::visit([&](const auto &x) { return genval(x); }, des.u);
  }

  mlir::Type genType(const Fortran::evaluate::DynamicType &dt) {
    if (dt.category() != Fortran::common::TypeCategory::Derived)
      return converter.genType(dt.category(), dt.kind());
    TODO(getLoc(), "genType Derived Type");
  }

  /// Lower a function reference
  template <typename A>
  ExtValue genFunctionRef(const Fortran::evaluate::FunctionRef<A> &funcRef) {
    if (!funcRef.GetType().has_value())
      fir::emitFatalError(getLoc(), "internal: a function must have a type");
    mlir::Type resTy = genType(*funcRef.GetType());
    return genProcedureRef(funcRef, {resTy});
  }

  /// Lower function call `funcRef` and return a reference to the resultant
  /// value. This is required for lowering expressions such as `f1(f2(v))`.
  template <typename A>
  ExtValue gen(const Fortran::evaluate::FunctionRef<A> &funcRef) {
    TODO(getLoc(), "gen FunctionRef<A>");
  }

  /// helper to detect statement functions
  static bool
  isStatementFunctionCall(const Fortran::evaluate::ProcedureRef &procRef) {
    if (const Fortran::semantics::Symbol *symbol = procRef.proc().GetSymbol())
      if (const auto *details =
              symbol->detailsIf<Fortran::semantics::SubprogramDetails>())
        return details->stmtFunction().has_value();
    return false;
  }

  /// Helper to package a Value and its properties into an ExtendedValue.
  static ExtValue toExtendedValue(mlir::Location loc, mlir::Value base,
                                  llvm::ArrayRef<mlir::Value> extents,
                                  llvm::ArrayRef<mlir::Value> lengths) {
    mlir::Type type = base.getType();
    if (type.isa<fir::BoxType>())
      return fir::BoxValue(base, /*lbounds=*/{}, lengths, extents);
    type = fir::unwrapRefType(type);
    if (type.isa<fir::BoxType>())
      return fir::MutableBoxValue(base, lengths, /*mutableProperties*/ {});
    if (auto seqTy = type.dyn_cast<fir::SequenceType>()) {
      if (seqTy.getDimension() != extents.size())
        fir::emitFatalError(loc, "incorrect number of extents for array");
      if (seqTy.getEleTy().isa<fir::CharacterType>()) {
        if (lengths.empty())
          fir::emitFatalError(loc, "missing length for character");
        assert(lengths.size() == 1);
        return fir::CharArrayBoxValue(base, lengths[0], extents);
      }
      return fir::ArrayBoxValue(base, extents);
    }
    if (type.isa<fir::CharacterType>()) {
      if (lengths.empty())
        fir::emitFatalError(loc, "missing length for character");
      assert(lengths.size() == 1);
      return fir::CharBoxValue(base, lengths[0]);
    }
    return base;
  }

  // Find the argument that corresponds to the host associations.
  // Verify some assumptions about how the signature was built here.
  [[maybe_unused]] static unsigned findHostAssocTuplePos(mlir::FuncOp fn) {
    // Scan the argument list from last to first as the host associations are
    // appended for now.
    for (unsigned i = fn.getNumArguments(); i > 0; --i)
      if (fn.getArgAttr(i - 1, fir::getHostAssocAttrName())) {
        // Host assoc tuple must be last argument (for now).
        assert(i == fn.getNumArguments() && "tuple must be last");
        return i - 1;
      }
    llvm_unreachable("anyFuncArgsHaveAttr failed");
  }

  /// Lower a non-elemental procedure reference and read allocatable and pointer
  /// results into normal values.
  ExtValue genProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                           llvm::Optional<mlir::Type> resultType) {
    ExtValue res = genRawProcedureRef(procRef, resultType);
    return res;
  }

  /// Given a call site for which the arguments were already lowered, generate
  /// the call and return the result. This function deals with explicit result
  /// allocation and lowering if needed. It also deals with passing the host
  /// link to internal procedures.
  ExtValue genCallOpAndResult(Fortran::lower::CallerInterface &caller,
                              mlir::FunctionType callSiteType,
                              llvm::Optional<mlir::Type> resultType) {
    mlir::Location loc = getLoc();
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;
    // Handle cases where caller must allocate the result or a fir.box for it.
    bool mustPopSymMap = false;
    if (caller.mustMapInterfaceSymbols()) {
      symMap.pushScope();
      mustPopSymMap = true;
      Fortran::lower::mapCallInterfaceSymbols(converter, caller, symMap);
    }
    // If this is an indirect call, retrieve the function address. Also retrieve
    // the result length if this is a character function (note that this length
    // will be used only if there is no explicit length in the local interface).
    mlir::Value funcPointer;
    mlir::Value charFuncPointerLength;
    if (caller.getIfIndirectCallSymbol()) {
      TODO(loc, "genCallOpAndResult indirect call");
    }

    mlir::IndexType idxTy = builder.getIndexType();
    auto lowerSpecExpr = [&](const auto &expr) -> mlir::Value {
      return builder.createConvert(
          loc, idxTy, fir::getBase(converter.genExprValue(expr, stmtCtx)));
    };
    llvm::SmallVector<mlir::Value> resultLengths;
    auto allocatedResult = [&]() -> llvm::Optional<ExtValue> {
      llvm::SmallVector<mlir::Value> extents;
      llvm::SmallVector<mlir::Value> lengths;
      if (!caller.callerAllocateResult())
        return {};
      mlir::Type type = caller.getResultStorageType();
      if (type.isa<fir::SequenceType>())
        caller.walkResultExtents([&](const Fortran::lower::SomeExpr &e) {
          extents.emplace_back(lowerSpecExpr(e));
        });
      caller.walkResultLengths([&](const Fortran::lower::SomeExpr &e) {
        lengths.emplace_back(lowerSpecExpr(e));
      });

      // Result length parameters should not be provided to box storage
      // allocation and save_results, but they are still useful information to
      // keep in the ExtendedValue if non-deferred.
      if (!type.isa<fir::BoxType>()) {
        if (fir::isa_char(fir::unwrapSequenceType(type)) && lengths.empty()) {
          // Calling an assumed length function. This is only possible if this
          // is a call to a character dummy procedure.
          if (!charFuncPointerLength)
            fir::emitFatalError(loc, "failed to retrieve character function "
                                     "length while calling it");
          lengths.push_back(charFuncPointerLength);
        }
        resultLengths = lengths;
      }

      if (!extents.empty() || !lengths.empty()) {
        TODO(loc, "genCallOpResult extents and length");
      }
      mlir::Value temp =
          builder.createTemporary(loc, type, ".result", extents, resultLengths);
      return toExtendedValue(loc, temp, extents, lengths);
    }();

    if (mustPopSymMap)
      symMap.popScope();

    // Place allocated result or prepare the fir.save_result arguments.
    mlir::Value arrayResultShape;
    if (allocatedResult) {
      if (std::optional<Fortran::lower::CallInterface<
              Fortran::lower::CallerInterface>::PassedEntity>
              resultArg = caller.getPassedResult()) {
        if (resultArg->passBy == PassBy::AddressAndLength)
          caller.placeAddressAndLengthInput(*resultArg,
                                            fir::getBase(*allocatedResult),
                                            fir::getLen(*allocatedResult));
        else if (resultArg->passBy == PassBy::BaseAddress)
          caller.placeInput(*resultArg, fir::getBase(*allocatedResult));
        else
          fir::emitFatalError(
              loc, "only expect character scalar result to be passed by ref");
      } else {
        assert(caller.mustSaveResult());
        arrayResultShape = allocatedResult->match(
            [&](const fir::CharArrayBoxValue &) {
              return builder.createShape(loc, *allocatedResult);
            },
            [&](const fir::ArrayBoxValue &) {
              return builder.createShape(loc, *allocatedResult);
            },
            [&](const auto &) { return mlir::Value{}; });
      }
    }

    // In older Fortran, procedure argument types are inferred. This may lead
    // different view of what the function signature is in different locations.
    // Casts are inserted as needed below to accommodate this.

    // The mlir::FuncOp type prevails, unless it has a different number of
    // arguments which can happen in legal program if it was passed as a dummy
    // procedure argument earlier with no further type information.
    mlir::SymbolRefAttr funcSymbolAttr;
    bool addHostAssociations = false;
    if (!funcPointer) {
      mlir::FunctionType funcOpType = caller.getFuncOp().getType();
      mlir::SymbolRefAttr symbolAttr =
          builder.getSymbolRefAttr(caller.getMangledName());
      if (callSiteType.getNumResults() == funcOpType.getNumResults() &&
          callSiteType.getNumInputs() + 1 == funcOpType.getNumInputs() &&
          fir::anyFuncArgsHaveAttr(caller.getFuncOp(),
                                   fir::getHostAssocAttrName())) {
        // The number of arguments is off by one, and we're lowering a function
        // with host associations. Modify call to include host associations
        // argument by appending the value at the end of the operands.
        assert(funcOpType.getInput(findHostAssocTuplePos(caller.getFuncOp())) ==
               converter.hostAssocTupleValue().getType());
        addHostAssociations = true;
      }
      if (!addHostAssociations &&
          (callSiteType.getNumResults() != funcOpType.getNumResults() ||
           callSiteType.getNumInputs() != funcOpType.getNumInputs())) {
        // Deal with argument number mismatch by making a function pointer so
        // that function type cast can be inserted. Do not emit a warning here
        // because this can happen in legal program if the function is not
        // defined here and it was first passed as an argument without any more
        // information.
        funcPointer =
            builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
      } else if (callSiteType.getResults() != funcOpType.getResults()) {
        // Implicit interface result type mismatch are not standard Fortran, but
        // some compilers are not complaining about it.  The front end is not
        // protecting lowering from this currently. Support this with a
        // discouraging warning.
        LLVM_DEBUG(mlir::emitWarning(
            loc, "a return type mismatch is not standard compliant and may "
                 "lead to undefined behavior."));
        // Cast the actual function to the current caller implicit type because
        // that is the behavior we would get if we could not see the definition.
        funcPointer =
            builder.create<fir::AddrOfOp>(loc, funcOpType, symbolAttr);
      } else {
        funcSymbolAttr = symbolAttr;
      }
    }

    mlir::FunctionType funcType =
        funcPointer ? callSiteType : caller.getFuncOp().getType();
    llvm::SmallVector<mlir::Value> operands;
    // First operand of indirect call is the function pointer. Cast it to
    // required function type for the call to handle procedures that have a
    // compatible interface in Fortran, but that have different signatures in
    // FIR.
    if (funcPointer) {
      operands.push_back(
          funcPointer.getType().isa<fir::BoxProcType>()
              ? builder.create<fir::BoxAddrOp>(loc, funcType, funcPointer)
              : builder.createConvert(loc, funcType, funcPointer));
    }

    // Deal with potential mismatches in arguments types. Passing an array to a
    // scalar argument should for instance be tolerated here.
    bool callingImplicitInterface = caller.canBeCalledViaImplicitInterface();
    for (auto [fst, snd] :
         llvm::zip(caller.getInputs(), funcType.getInputs())) {
      // When passing arguments to a procedure that can be called an implicit
      // interface, allow character actual arguments to be passed to dummy
      // arguments of any type and vice versa
      mlir::Value cast;
      auto *context = builder.getContext();
      if (snd.isa<fir::BoxProcType>() &&
          fst.getType().isa<mlir::FunctionType>()) {
        auto funcTy = mlir::FunctionType::get(context, llvm::None, llvm::None);
        auto boxProcTy = builder.getBoxProcType(funcTy);
        if (mlir::Value host = argumentHostAssocs(converter, fst)) {
          cast = builder.create<fir::EmboxProcOp>(
              loc, boxProcTy, llvm::ArrayRef<mlir::Value>{fst, host});
        } else {
          cast = builder.create<fir::EmboxProcOp>(loc, boxProcTy, fst);
        }
      } else {
        cast = builder.convertWithSemantics(loc, snd, fst,
                                            callingImplicitInterface);
      }
      operands.push_back(cast);
    }

    // Add host associations as necessary.
    if (addHostAssociations)
      operands.push_back(converter.hostAssocTupleValue());

    auto call = builder.create<fir::CallOp>(loc, funcType.getResults(),
                                            funcSymbolAttr, operands);

    if (caller.mustSaveResult())
      builder.create<fir::SaveResultOp>(
          loc, call.getResult(0), fir::getBase(allocatedResult.getValue()),
          arrayResultShape, resultLengths);

    if (allocatedResult) {
      allocatedResult->match(
          [&](const fir::MutableBoxValue &box) {
            if (box.isAllocatable()) {
              TODO(loc, "allocatedResult for allocatable");
            }
          },
          [](const auto &) {});
      return *allocatedResult;
    }

    if (!resultType.hasValue())
      return mlir::Value{}; // subroutine call
    // For now, Fortran return values are implemented with a single MLIR
    // function return value.
    assert(call.getNumResults() == 1 &&
           "Expected exactly one result in FUNCTION call");
    return call.getResult(0);
  }

  /// Like genExtAddr, but ensure the address returned is a temporary even if \p
  /// expr is variable inside parentheses.
  ExtValue genTempExtAddr(const Fortran::lower::SomeExpr &expr) {
    // In general, genExtAddr might not create a temp for variable inside
    // parentheses to avoid creating array temporary in sub-expressions. It only
    // ensures the sub-expression is not re-associated with other parts of the
    // expression. In the call semantics, there is a difference between expr and
    // variable (see R1524). For expressions, a variable storage must not be
    // argument associated since it could be modified inside the call, or the
    // variable could also be modified by other means during the call.
    if (!isParenthesizedVariable(expr))
      return genExtAddr(expr);
    mlir::Location loc = getLoc();
    if (expr.Rank() > 0)
      TODO(loc, "genTempExtAddr array");
    return genExtValue(expr).match(
        [&](const fir::CharBoxValue &boxChar) -> ExtValue {
          TODO(loc, "genTempExtAddr CharBoxValue");
        },
        [&](const fir::UnboxedValue &v) -> ExtValue {
          mlir::Type type = v.getType();
          mlir::Value value = v;
          if (fir::isa_ref_type(type))
            value = builder.create<fir::LoadOp>(loc, value);
          mlir::Value temp = builder.createTemporary(loc, value.getType());
          builder.create<fir::StoreOp>(loc, value, temp);
          return temp;
        },
        [&](const fir::BoxValue &x) -> ExtValue {
          // Derived type scalar that may be polymorphic.
          assert(!x.hasRank() && x.isDerived());
          if (x.isDerivedWithLengthParameters())
            fir::emitFatalError(
                loc, "making temps for derived type with length parameters");
          // TODO: polymorphic aspects should be kept but for now the temp
          // created always has the declared type.
          mlir::Value var =
              fir::getBase(fir::factory::readBoxValue(builder, loc, x));
          auto value = builder.create<fir::LoadOp>(loc, var);
          mlir::Value temp = builder.createTemporary(loc, value.getType());
          builder.create<fir::StoreOp>(loc, value, temp);
          return temp;
        },
        [&](const auto &) -> ExtValue {
          fir::emitFatalError(loc, "expr is not a scalar value");
        });
  }

  /// Helper structure to track potential copy-in of non contiguous variable
  /// argument into a contiguous temp. It is used to deallocate the temp that
  /// may have been created as well as to the copy-out from the temp to the
  /// variable after the call.
  struct CopyOutPair {
    ExtValue var;
    ExtValue temp;
    // Flag to indicate if the argument may have been modified by the
    // callee, in which case it must be copied-out to the variable.
    bool argMayBeModifiedByCall;
    // Optional boolean value that, if present and false, prevents
    // the copy-out and temp deallocation.
    llvm::Optional<mlir::Value> restrictCopyAndFreeAtRuntime;
  };
  using CopyOutPairs = llvm::SmallVector<CopyOutPair, 4>;

  /// Helper to read any fir::BoxValue into other fir::ExtendedValue categories
  /// not based on fir.box.
  /// This will lose any non contiguous stride information and dynamic type and
  /// should only be called if \p exv is known to be contiguous or if its base
  /// address will be replaced by a contiguous one. If \p exv is not a
  /// fir::BoxValue, this is a no-op.
  ExtValue readIfBoxValue(const ExtValue &exv) {
    if (const auto *box = exv.getBoxOf<fir::BoxValue>())
      return fir::factory::readBoxValue(builder, getLoc(), *box);
    return exv;
  }

  /// Lower a non-elemental procedure reference.
  ExtValue genRawProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                              llvm::Optional<mlir::Type> resultType) {
    mlir::Location loc = getLoc();
    if (isElementalProcWithArrayArgs(procRef))
      fir::emitFatalError(loc, "trying to lower elemental procedure with array "
                               "arguments as normal procedure");
    if (const Fortran::evaluate::SpecificIntrinsic *intrinsic =
            procRef.proc().GetSpecificIntrinsic())
      return genIntrinsicRef(procRef, *intrinsic, resultType);

    if (isStatementFunctionCall(procRef))
      TODO(loc, "Lower statement function call");

    Fortran::lower::CallerInterface caller(procRef, converter);
    using PassBy = Fortran::lower::CallerInterface::PassEntityBy;

    llvm::SmallVector<fir::MutableBoxValue> mutableModifiedByCall;
    // List of <var, temp> where temp must be copied into var after the call.
    CopyOutPairs copyOutPairs;

    mlir::FunctionType callSiteType = caller.genFunctionType();

    // Lower the actual arguments and map the lowered values to the dummy
    // arguments.
    for (const Fortran::lower::CallInterface<
             Fortran::lower::CallerInterface>::PassedEntity &arg :
         caller.getPassedArguments()) {
      const auto *actual = arg.entity;
      mlir::Type argTy = callSiteType.getInput(arg.firArgument);
      if (!actual) {
        // Optional dummy argument for which there is no actual argument.
        caller.placeInput(arg, builder.create<fir::AbsentOp>(loc, argTy));
        continue;
      }
      const auto *expr = actual->UnwrapExpr();
      if (!expr)
        TODO(loc, "assumed type actual argument lowering");

      if (arg.passBy == PassBy::Value) {
        ExtValue argVal = genval(*expr);
        if (!fir::isUnboxedValue(argVal))
          fir::emitFatalError(
              loc, "internal error: passing non trivial value by value");
        caller.placeInput(arg, fir::getBase(argVal));
        continue;
      }

      if (arg.passBy == PassBy::MutableBox) {
        TODO(loc, "arg passby MutableBox");
      }
      const bool actualArgIsVariable = Fortran::evaluate::IsVariable(*expr);
      if (arg.passBy == PassBy::BaseAddress || arg.passBy == PassBy::BoxChar) {
        auto argAddr = [&]() -> ExtValue {
          ExtValue baseAddr;
          if (actualArgIsVariable && arg.isOptional()) {
            if (Fortran::evaluate::IsAllocatableOrPointerObject(
                    *expr, converter.getFoldingContext())) {
              TODO(loc, "Allocatable or pointer argument");
            }
            if (const Fortran::semantics::Symbol *wholeSymbol =
                    Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(
                        *expr))
              if (Fortran::semantics::IsOptional(*wholeSymbol)) {
                TODO(loc, "procedureref optional arg");
              }
            // Fall through: The actual argument can safely be
            // copied-in/copied-out without any care if needed.
          }
          if (actualArgIsVariable && expr->Rank() > 0) {
            TODO(loc, "procedureref arrays");
          }
          // Actual argument is a non optional/non pointer/non allocatable
          // scalar.
          if (actualArgIsVariable)
            return genExtAddr(*expr);
          // Actual argument is not a variable. Make sure a variable address is
          // not passed.
          return genTempExtAddr(*expr);
        }();
        // Scalar and contiguous expressions may be lowered to a fir.box,
        // either to account for potential polymorphism, or because lowering
        // did not account for some contiguity hints.
        // Here, polymorphism does not matter (an entity of the declared type
        // is passed, not one of the dynamic type), and the expr is known to
        // be simply contiguous, so it is safe to unbox it and pass the
        // address without making a copy.
        argAddr = readIfBoxValue(argAddr);

        if (arg.passBy == PassBy::BaseAddress) {
          caller.placeInput(arg, fir::getBase(argAddr));
        } else {
          TODO(loc, "procedureref PassBy::BoxChar");
        }
      } else if (arg.passBy == PassBy::Box) {
        // Before lowering to an address, handle the allocatable/pointer actual
        // argument to optional fir.box dummy. It is legal to pass
        // unallocated/disassociated entity to an optional. In this case, an
        // absent fir.box must be created instead of a fir.box with a null value
        // (Fortran 2018 15.5.2.12 point 1).
        if (arg.isOptional() && Fortran::evaluate::IsAllocatableOrPointerObject(
                                    *expr, converter.getFoldingContext())) {
          TODO(loc, "optional allocatable or pointer argument");
        } else {
          // Make sure a variable address is only passed if the expression is
          // actually a variable.
          mlir::Value box =
              actualArgIsVariable
                  ? builder.createBox(loc, genBoxArg(*expr))
                  : builder.createBox(getLoc(), genTempExtAddr(*expr));
          caller.placeInput(arg, box);
        }
      } else if (arg.passBy == PassBy::AddressAndLength) {
        ExtValue argRef = genExtAddr(*expr);
        caller.placeAddressAndLengthInput(arg, fir::getBase(argRef),
                                          fir::getLen(argRef));
      } else if (arg.passBy == PassBy::CharProcTuple) {
        TODO(loc, "procedureref CharProcTuple");
      } else {
        TODO(loc, "pass by value in non elemental function call");
      }
    }

    ExtValue result = genCallOpAndResult(caller, callSiteType, resultType);

    // // Copy-out temps that were created for non contiguous variable arguments
    // if
    // // needed.
    // for (const auto &copyOutPair : copyOutPairs)
    //   genCopyOut(copyOutPair);

    return result;
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::FunctionRef<A> &funcRef) {
    ExtValue result = genFunctionRef(funcRef);
    if (result.rank() == 0 && fir::isa_ref_type(fir::getBase(result).getType()))
      return genLoad(result);
    return result;
  }

  ExtValue genval(const Fortran::evaluate::ProcedureRef &procRef) {
    llvm::Optional<mlir::Type> resTy;
    if (procRef.hasAlternateReturns())
      resTy = builder.getIndexType();
    return genProcedureRef(procRef, resTy);
  }

  /// Generate a call to an intrinsic function.
  ExtValue
  genIntrinsicRef(const Fortran::evaluate::ProcedureRef &procRef,
                  const Fortran::evaluate::SpecificIntrinsic &intrinsic,
                  llvm::Optional<mlir::Type> resultType) {
    llvm::SmallVector<ExtValue> operands;

    llvm::StringRef name = intrinsic.name;
    mlir::Location loc = getLoc();

    const Fortran::lower::IntrinsicArgumentLoweringRules *argLowering =
        Fortran::lower::getIntrinsicArgumentLowering(name);
    for (const auto &[arg, dummy] :
         llvm::zip(procRef.arguments(),
                   intrinsic.characteristics.value().dummyArguments)) {
      auto *expr = Fortran::evaluate::UnwrapExpr<Fortran::lower::SomeExpr>(arg);
      if (!expr) {
        // Absent optional.
        operands.emplace_back(Fortran::lower::getAbsentIntrinsicArgument());
        continue;
      }
      if (!argLowering) {
        // No argument lowering instruction, lower by value.
        operands.emplace_back(genval(*expr));
        continue;
      }
      // Ad-hoc argument lowering handling.
      Fortran::lower::ArgLoweringRule argRules =
          Fortran::lower::lowerIntrinsicArgumentAs(loc, *argLowering,
                                                   dummy.name);
      switch (argRules.lowerAs) {
      case Fortran::lower::LowerIntrinsicArgAs::Value:
        operands.emplace_back(genval(*expr));
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Addr:
        TODO(getLoc(), "argument lowering for Addr");
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Box:
        TODO(getLoc(), "argument lowering for Box");
        continue;
      case Fortran::lower::LowerIntrinsicArgAs::Inquired:
        TODO(getLoc(), "argument lowering for Inquired");
        continue;
      }
      llvm_unreachable("bad switch");
    }
    // Let the intrinsic library lower the intrinsic procedure call
    return Fortran::lower::genIntrinsicCall(builder, getLoc(), name, resultType,
                                            operands);
  }

  template <typename A>
  ExtValue genval(const Fortran::evaluate::Expr<A> &x) {
    if (isScalar(x))
      return std::visit([&](const auto &e) { return genval(e); }, x.u);
    TODO(getLoc(), "genval Expr<A> arrays");
  }

  /// Helper to detect Transformational function reference.
  template <typename T>
  bool isTransformationalRef(const T &) {
    return false;
  }
  template <typename T>
  bool isTransformationalRef(const Fortran::evaluate::FunctionRef<T> &funcRef) {
    return !funcRef.IsElemental() && funcRef.Rank();
  }
  template <typename T>
  bool isTransformationalRef(Fortran::evaluate::Expr<T> expr) {
    return std::visit([&](const auto &e) { return isTransformationalRef(e); },
                      expr.u);
  }

  template <typename A>
  ExtValue gen(const Fortran::evaluate::Expr<A> &x) {
    // Whole array symbols or components, and results of transformational
    // functions already have a storage and the scalar expression lowering path
    // is used to not create a new temporary storage.
    if (isScalar(x) ||
        Fortran::evaluate::UnwrapWholeSymbolOrComponentDataRef(x) ||
        isTransformationalRef(x))
      return std::visit([&](const auto &e) { return genref(e); }, x.u);
    TODO(getLoc(), "gen Expr non-scalar");
  }

  template <typename A>
  bool isScalar(const A &x) {
    return x.Rank() == 0;
  }

  template <int KIND>
  ExtValue genval(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                      Fortran::common::TypeCategory::Logical, KIND>> &exp) {
    return std::visit([&](const auto &e) { return genval(e); }, exp.u);
  }

  using RefSet =
      std::tuple<Fortran::evaluate::ComplexPart, Fortran::evaluate::Substring,
                 Fortran::evaluate::DataRef, Fortran::evaluate::Component,
                 Fortran::evaluate::ArrayRef, Fortran::evaluate::CoarrayRef,
                 Fortran::semantics::SymbolRef>;
  template <typename A>
  static constexpr bool inRefSet = Fortran::common::HasMember<A, RefSet>;

  template <typename A, typename = std::enable_if_t<inRefSet<A>>>
  ExtValue genref(const A &a) {
    return gen(a);
  }
  template <typename A>
  ExtValue genref(const A &a) {
    mlir::Type storageType = converter.genType(toEvExpr(a));
    return placeScalarValueInMemory(builder, getLoc(), genval(a), storageType);
  }

  template <typename A, template <typename> typename T,
            typename B = std::decay_t<T<A>>,
            std::enable_if_t<
                std::is_same_v<B, Fortran::evaluate::Expr<A>> ||
                    std::is_same_v<B, Fortran::evaluate::Designator<A>> ||
                    std::is_same_v<B, Fortran::evaluate::FunctionRef<A>>,
                bool> = true>
  ExtValue genref(const T<A> &x) {
    return gen(x);
  }

private:
  mlir::Location location;
  Fortran::lower::AbstractConverter &converter;
  fir::FirOpBuilder &builder;
  Fortran::lower::StatementContext &stmtCtx;
  Fortran::lower::SymMap &symMap;
  bool useBoxArg = false; // expression lowered as argument
};
} // namespace

fir::ExtendedValue Fortran::lower::createSomeExtendedExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.genval(expr);
}

fir::ExtendedValue Fortran::lower::createSomeExtendedAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap,
    Fortran::lower::StatementContext &stmtCtx) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "address: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap, stmtCtx}.gen(expr);
}

mlir::Value Fortran::lower::createSubroutineCall(
    AbstractConverter &converter, const evaluate::ProcedureRef &call,
    SymMap &symMap, StatementContext &stmtCtx) {
  mlir::Location loc = converter.getCurrentLocation();

  // Simple subroutine call, with potential alternate return.
  auto res = Fortran::lower::createSomeExtendedExpression(
      loc, converter, toEvExpr(call), symMap, stmtCtx);
  return fir::getBase(res);
}
