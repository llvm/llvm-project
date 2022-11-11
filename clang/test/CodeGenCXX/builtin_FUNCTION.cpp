// RUN: %clang_cc1 -std=c++2a -fblocks %s -triple %itanium_abi_triple -emit-llvm -o %t.ll
// RUN: FileCheck --input-file %t.ll %s

namespace test_func {

constexpr const char *test_default_arg(const char *f = __builtin_FUNCTION()) {
  return f;
}
// CHECK: @[[EMPTY_STR:.+]] = private unnamed_addr constant [1 x i8] zeroinitializer

// CHECK: @_ZN9test_func6globalE = {{(dso_local )?}}global ptr @[[EMPTY_STR]]
const char *global = test_default_arg();

// CHECK: @_ZN9test_func10global_twoE = {{(dso_local )?}}global ptr @[[EMPTY_STR]]
const char *global_two = __builtin_FUNCTION();

const char * const global_three = test_default_arg();

// CHECK: @[[STR_ONE:.+]] = private unnamed_addr constant [14 x i8] c"test_func_one\00"
// CHECK: @[[STR_TWO:.+]] = private unnamed_addr constant [14 x i8] c"test_func_two\00"
// CHECK: @[[STR_THREE:.+]] = private unnamed_addr constant [20 x i8] c"do_default_arg_test\00"

// CHECK: define {{(dso_local )?}}noundef ptr @_ZN9test_func13test_func_oneEv()
// CHECK: ret ptr @[[STR_ONE]]
const char *test_func_one() {
  return __builtin_FUNCTION();
}

// CHECK: define {{(dso_local )?}}noundef ptr @_ZN9test_func13test_func_twoEv()
// CHECK: ret ptr @[[STR_TWO]]
const char *test_func_two() {
  return __builtin_FUNCTION();
}

// CHECK: define {{(dso_local )?}}void @_ZN9test_func19do_default_arg_testEv()
// CHECK: %call = call noundef ptr @_ZN9test_func16test_default_argEPKc(ptr noundef @[[STR_THREE]])
void do_default_arg_test() {
  test_default_arg();
}

} // namespace test_func
