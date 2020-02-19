#include "testing.h"
#include "../../lib/Evaluate/host.h"
#include "../../lib/Evaluate/intrinsics-library-templates.h"
#include "flang/Evaluate/call.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/tools.h"
#include <tuple>

using namespace Fortran::evaluate;

// helper to call functions on all types from tuple
template<typename... T> struct RunOnTypes {};
template<typename Test, typename... T>
struct RunOnTypes<Test, std::tuple<T...>> {
  static void Run() { (..., Test::template Run<T>()); }
};

// test for fold.h GetScalarConstantValue function
struct TestGetScalarConstantValue {
  template<typename T> static void Run() {
    Expr<T> exprFullyTyped{Constant<T>{Scalar<T>{}}};
    Expr<SomeKind<T::category>> exprSomeKind{exprFullyTyped};
    Expr<SomeType> exprSomeType{exprSomeKind};
    TEST(GetScalarConstantValue<T>(exprFullyTyped).has_value());
    TEST(GetScalarConstantValue<T>(exprSomeKind).has_value());
    TEST(GetScalarConstantValue<T>(exprSomeType).has_value());
  }
};

template<typename T>
static FunctionRef<T> CreateIntrinsicElementalCall(
    const std::string &name, const Expr<T> &arg) {
  Fortran::semantics::Attrs attrs;
  attrs.set(Fortran::semantics::Attr::ELEMENTAL);
  ActualArguments args{ActualArgument{AsGenericExpr(arg)}};
  ProcedureDesignator intrinsic{
      SpecificIntrinsic{name, T::GetType(), 0, attrs}};
  return FunctionRef<T>{std::move(intrinsic), std::move(args)};
}

// Test flushSubnormalsToZero when folding with host runtime.
// Subnormal value flushing on host is handle in host.cpp
// HostFloatingPointEnvironment::SetUpHostFloatingPointEnvironment

// Dummy host runtime functions where subnormal flushing matters
float SubnormalFlusher1(float f) {  // given f is subnormal
  return 2.3 * f;  // returns 0 if subnormal arguments are flushed to zero
}

float SubnormalFlusher2(float f) {  // given f/2 is subnormal
  return f / 2.3;  // returns 0 if subnormal
}

void TestHostRuntimeSubnormalFlushing() {
  using R4 = Type<TypeCategory::Real, 4>;
  if constexpr (std::is_same_v<host::HostType<R4>, float>) {
    Fortran::parser::CharBlock src;
    Fortran::parser::ContextualMessages messages{src, nullptr};
    Fortran::common::IntrinsicTypeDefaultKinds defaults;
    auto intrinsics{Fortran::evaluate::IntrinsicProcTable::Configure(defaults)};
    FoldingContext flushingContext{
        messages, defaults, intrinsics, defaultRounding, true};
    FoldingContext noFlushingContext{
        messages, defaults, intrinsics, defaultRounding, false};

    HostIntrinsicProceduresLibrary lib;
    lib.AddProcedure(HostRuntimeIntrinsicProcedure{
        "flusher_test1", SubnormalFlusher1, true});
    lib.AddProcedure(HostRuntimeIntrinsicProcedure{
        "flusher_test2", SubnormalFlusher2, true});

    // Test subnormal argument flushing
    if (auto callable{
            lib.GetHostProcedureWrapper<Scalar, R4, R4>("flusher_test1")}) {
      // Biggest IEEE 32bits subnormal power of two
      host::HostType<R4> input1{5.87747175411144e-39};
      const Scalar<R4> x1{host::CastHostToFortran<R4>(input1)};
      Scalar<R4> y1Flushing{callable.value()(flushingContext, x1)};
      Scalar<R4> y1NoFlushing{callable.value()(noFlushingContext, x1)};
      TEST(y1Flushing.IsZero());
      TEST(!y1NoFlushing.IsZero());
    } else {
      TEST(false);
    }
    // Test subnormal result flushing
    if (auto callable{
            lib.GetHostProcedureWrapper<Scalar, R4, R4>("flusher_test2")}) {
      // Smallest (positive) non-subnormal IEEE 32 bit float value
      host::HostType<R4> input2{1.1754944e-38};
      const Scalar<R4> x2{host::CastHostToFortran<R4>(input2)};
      Scalar<R4> y2Flushing{callable.value()(flushingContext, x2)};
      Scalar<R4> y2NoFlushing{callable.value()(noFlushingContext, x2)};
      TEST(y2Flushing.IsZero());
      TEST(!y2NoFlushing.IsZero());
    } else {
      TEST(false);
    }
  } else {
    TEST(false);  // Cannot run this test on the host
  }
}

int main() {
  RunOnTypes<TestGetScalarConstantValue, AllIntrinsicTypes>::Run();
  TestHostRuntimeSubnormalFlushing();
  return testing::Complete();
}
