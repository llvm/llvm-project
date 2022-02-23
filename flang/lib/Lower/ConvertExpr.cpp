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
#include "flang/Evaluate/real.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/SymbolMap.h"
#include "flang/Lower/Todo.h"
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
namespace {

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ScalarExprLowering {
public:
  using ExtValue = fir::ExtendedValue;

  explicit ScalarExprLowering(mlir::Location loc,
                              Fortran::lower::AbstractConverter &converter,
                              Fortran::lower::SymMap &symMap)
      : location{loc}, converter{converter},
        builder{converter.getFirOpBuilder()}, symMap{symMap} {}

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
    TODO(getLoc(), "genval ComplexConstructor");
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
      TODO(getLoc(), "genval complex constant");
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

  template <typename A>
  ExtValue genval(const Fortran::evaluate::FunctionRef<A> &funcRef) {
    ExtValue result = genFunctionRef(funcRef);
    if (result.rank() == 0 && fir::isa_ref_type(fir::getBase(result).getType()))
      return genLoad(result);
    return result;
  }

  ExtValue genval(const Fortran::evaluate::ProcedureRef &procRef) {
    TODO(getLoc(), "genval ProcedureRef");
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

  /// Lower a non-elemental procedure reference.
  // TODO: Handle read allocatable and pointer results.
  ExtValue genProcedureRef(const Fortran::evaluate::ProcedureRef &procRef,
                           llvm::Optional<mlir::Type> resultType) {
    ExtValue res = genRawProcedureRef(procRef, resultType);
    return res;
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

    return {};
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
  Fortran::lower::SymMap &symMap;
};
} // namespace

fir::ExtendedValue Fortran::lower::createSomeExtendedExpression(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "expr: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap}.genval(expr);
}

fir::ExtendedValue Fortran::lower::createSomeExtendedAddress(
    mlir::Location loc, Fortran::lower::AbstractConverter &converter,
    const Fortran::lower::SomeExpr &expr, Fortran::lower::SymMap &symMap) {
  LLVM_DEBUG(expr.AsFortran(llvm::dbgs() << "address: ") << '\n');
  return ScalarExprLowering{loc, converter, symMap}.gen(expr);
}
