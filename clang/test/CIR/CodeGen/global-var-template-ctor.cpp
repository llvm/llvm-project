// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

// Regression test: C++14 variable templates with non-constexpr
// constructors must not crash CIR with "ctorRegion failed to
// verify constraint: region with at most 1 blocks".

template<int N> class FixedInt {
public:
  static const int value = N;
  operator int() const { return value; }
  FixedInt() {}
};

template<int N>
static const FixedInt<N> fix{};

int test() {
  return fix<1> + fix<2>;
}

// make sure we only emit the initializer 1x, even if it is used 2x.
int get_value();
template<int N> int dyn = get_value();

int test_multi_use() {
  return dyn<1> + dyn<1>;
}
// CIR-DAG: cir.global "private" internal  dso_local @_ZL3fixILi1EE = #cir.zero : !rec_FixedInt3C13E
// CIR-DAG: cir.global "private" internal  dso_local @_ZL3fixILi2EE = #cir.zero : !rec_FixedInt3C23E
// CIR-DAG: cir.global linkonce_odr comdat dynamic_init_guard<"_ZGV3dynILi1EE"> @_Z3dynILi1EE = #cir.int<0> : !s32i 

// LLVM-DAG: @_ZL3fixILi1EE = internal global %{{.*}}FixedInt{{.*}} zeroinitializer
// LLVM-DAG: @_ZL3fixILi2EE = internal global %{{.*}}FixedInt{{.*}} zeroinitializer
// LLVM-DAG: @_ZGV3dynILi1EE = linkonce_odr global i64 0, comdat($_Z3dynILi1EE)

// Classic orderes this first for some reason, so we have to have a separate
// check line.
// OGCG-DAG: define {{.*}} @_Z4testv

// CIR-LABEL: cir.func comdat("_Z3dynILi1EE") {{.*}}@__cxx_global_var_init
// CIR:   cir.call @_Z9get_valuev()
// CIR-NOT: cir.call @_Z9get_valuev()

// LLVM-LABEL: define internal void @__cxx_global_var_init{{.*}}() {{.*}}comdat($_Z3dynILi1EE) {
// LLVM: call {{.*}}@_Z9get_valuev()
// LLVM-NOT: call {{.*}}@_Z9get_valuev()

// CIR-DAG: cir.func {{.*}} @_Z4testv
// LLVMCIR-DAG: define {{.*}} @_Z4testv
