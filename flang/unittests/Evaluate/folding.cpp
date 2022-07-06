#include "testing.h"
#include "../../lib/Evaluate/host.h"
#include "flang/Evaluate/call.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/intrinsics-library.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/target.h"
#include "flang/Evaluate/tools.h"
#include <tuple>

using namespace Fortran::evaluate;

// helper to call functions on all types from tuple
template <typename... T> struct RunOnTypes {};
template <typename Test, typename... T>
struct RunOnTypes<Test, std::tuple<T...>> {
  static void Run() { (..., Test::template Run<T>()); }
};

// test for fold.h GetScalarConstantValue function
struct TestGetScalarConstantValue {
  template <typename T> static void Run() {
    Expr<T> exprFullyTyped{Constant<T>{Scalar<T>{}}};
    Expr<SomeKind<T::category>> exprSomeKind{exprFullyTyped};
    Expr<SomeType> exprSomeType{exprSomeKind};
    TEST(GetScalarConstantValue<T>(exprFullyTyped).has_value());
    TEST(GetScalarConstantValue<T>(exprSomeKind).has_value());
    TEST(GetScalarConstantValue<T>(exprSomeType).has_value());
  }
};

template <typename T>
Scalar<T> CallHostRt(
    HostRuntimeWrapper func, FoldingContext &context, Scalar<T> x) {
  return GetScalarConstantValue<T>(
      func(context, {AsGenericExpr(Constant<T>{x})}))
      .value();
}

void TestHostRuntimeSubnormalFlushing() {
  using R4 = Type<TypeCategory::Real, 4>;
  if constexpr (std::is_same_v<host::HostType<R4>, float>) {
    Fortran::parser::CharBlock src;
    Fortran::parser::ContextualMessages messages{src, nullptr};
    Fortran::common::IntrinsicTypeDefaultKinds defaults;
    auto intrinsics{Fortran::evaluate::IntrinsicProcTable::Configure(defaults)};
    TargetCharacteristics flushingTargetCharacteristics;
    flushingTargetCharacteristics.set_areSubnormalsFlushedToZero(true);
    TargetCharacteristics noFlushingTargetCharacteristics;
    noFlushingTargetCharacteristics.set_areSubnormalsFlushedToZero(false);
    FoldingContext flushingContext{
        messages, defaults, intrinsics, flushingTargetCharacteristics};
    FoldingContext noFlushingContext{
        messages, defaults, intrinsics, noFlushingTargetCharacteristics};

    DynamicType r4{R4{}.GetType()};
    // Test subnormal argument flushing
    if (auto callable{GetHostRuntimeWrapper("log", r4, {r4})}) {
      // Biggest IEEE 32bits subnormal power of two
      const Scalar<R4> x1{Scalar<R4>::Word{0x00400000}};
      Scalar<R4> y1Flushing{CallHostRt<R4>(*callable, flushingContext, x1)};
      Scalar<R4> y1NoFlushing{CallHostRt<R4>(*callable, noFlushingContext, x1)};
      // We would expect y1Flushing to be NaN, but some libc logf implementation
      // "workaround" subnormal flushing by returning a constant negative
      // results for all subnormal values (-1.03972076416015625e2_4). In case of
      // flushing, the result should still be different than -88 +/- 2%.
      TEST(y1Flushing.IsInfinite() ||
          std::abs(host::CastFortranToHost<R4>(y1Flushing) + 88.) > 2);
      TEST(!y1NoFlushing.IsInfinite() &&
          std::abs(host::CastFortranToHost<R4>(y1NoFlushing) + 88.) < 2);
    } else {
      TEST(false);
    }
  } else {
    TEST(false); // Cannot run this test on the host
  }
}

int main() {
  RunOnTypes<TestGetScalarConstantValue, AllIntrinsicTypes>::Run();
  TestHostRuntimeSubnormalFlushing();
  return testing::Complete();
}
