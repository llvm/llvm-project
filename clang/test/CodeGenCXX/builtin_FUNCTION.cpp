// RUN: %clang_cc1 -std=c++2a -fblocks %s -triple %itanium_abi_triple -emit-llvm -o %t.ll
// RUN: FileCheck --input-file %t.ll %s

namespace test_func {

constexpr const char *test_default_arg(const char *f = __builtin_FUNCTION()) {
  return f;
}
// CHECK: @[[EMPTY_STR:.+]] = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

// CHECK: @_ZN9test_func6globalE = global i8* getelementptr inbounds ([1 x i8], [1 x i8]* @[[EMPTY_STR]], i32 0, i32 0), align 8
const char *global = test_default_arg();

// CHECK: @_ZN9test_func10global_twoE = global i8* getelementptr inbounds ([1 x i8], [1 x i8]* @[[EMPTY_STR]], i32 0, i32 0), align 8
const char *global_two = __builtin_FUNCTION();

const char * const global_three = test_default_arg();

// CHECK: @[[STR_ONE:.+]] = private unnamed_addr constant [14 x i8] c"test_func_one\00", align 1
// CHECK: @[[STR_TWO:.+]] = private unnamed_addr constant [14 x i8] c"test_func_two\00", align 1
// CHECK: @[[STR_THREE:.+]] = private unnamed_addr constant [20 x i8] c"do_default_arg_test\00", align 1

// CHECK: define i8* @_ZN9test_func13test_func_oneEv()
// CHECK: ret i8* getelementptr inbounds ([14 x i8], [14 x i8]* @[[STR_ONE]], i32 0, i32 0)
const char *test_func_one() {
  return __builtin_FUNCTION();
}

// CHECK: define i8* @_ZN9test_func13test_func_twoEv()
// CHECK: ret i8* getelementptr inbounds ([14 x i8], [14 x i8]* @[[STR_TWO]], i32 0, i32 0)
const char *test_func_two() {
  return __builtin_FUNCTION();
}

// CHECK: define void @_ZN9test_func19do_default_arg_testEv()
// CHECK: %call = call i8* @_ZN9test_func16test_default_argEPKc(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @[[STR_THREE]], i32 0, i32 0))
void do_default_arg_test() {
  test_default_arg();
}

} // namespace test_func
