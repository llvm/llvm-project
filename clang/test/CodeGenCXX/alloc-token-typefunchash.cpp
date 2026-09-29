// Test that the typefunchash modes add the name of the function containing the
// allocation to the !alloc_token metadata, and always emit metadata.
//
// RUN: %clang_cc1 -fsanitize=alloc-token -falloc-token-mode=typefunchash -triple x86_64-linux-gnu -std=c++20 -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s
// RUN: %clang_cc1 -fsanitize=alloc-token -falloc-token-mode=typefunchashpointersplit -triple x86_64-linux-gnu -std=c++20 -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

typedef __typeof(sizeof(int)) size_t;
extern "C" void *malloc(size_t size) __attribute__((malloc));

struct WithPtr {
  int a;
  char *buf;
};

struct Incomplete;

void *sink;

// Unknown or incomplete types use an empty type name.
// CHECK-LABEL: define {{.*}} @_Z9test_funcm(
// CHECK: call {{.*}} @malloc(i64 noundef 4){{.*}}, !alloc_token [[META_FUNC:![0-9]+]]
// CHECK: call {{.*}} @malloc(i64 noundef %{{.*}}){{.*}}, !alloc_token [[META_UNKNOWN:![0-9]+]]
// CHECK: call {{.*}} @malloc(i64 noundef 4){{.*}}, !alloc_token [[META_UNKNOWN]]
void test_func(size_t n) {
  sink = (int *)malloc(sizeof(int));
  sink = malloc(n);
  sink = (Incomplete *)malloc(4);
}

namespace ns {
struct S {
  void *method();
};
// CHECK-LABEL: define {{.*}} @_ZN2ns1S6methodEv(
// CHECK: call {{.*}} @_Znwm(i64 noundef 16){{.*}}, !alloc_token [[META_METHOD:![0-9]+]]
void *S::method() { return new WithPtr; }
} // namespace ns

template <typename T>
struct Tmpl {
  static void *alloc() { return new T; }
};

// CHECK-LABEL: define {{.*}} @_ZN4TmplIlE5allocEv(
// CHECK: call {{.*}} @_Znwm(i64 noundef 8){{.*}}, !alloc_token [[META_TMPL:![0-9]+]]
void test_template() { sink = Tmpl<long>::alloc(); }

// Lambdas use the name of their call operator.
// CHECK-LABEL: define {{.*}} @"_ZZ11test_lambdavENK{{.*}}clEv"(
// CHECK: call {{.*}} @_Znwm(i64 noundef 4){{.*}}, !alloc_token [[META_LAMBDA:![0-9]+]]
void test_lambda() {
  auto L = [] { return new int; };
  sink = L();
}

// Runtime use of the builtin, which is not a constant expression in this mode.
// CHECK-LABEL: define {{.*}} @_Z12test_builtinv(
// CHECK: call i64 @llvm.alloc.token.id.i64(metadata [[META_BUILTIN:![0-9]+]])
unsigned long test_builtin() { return __builtin_infer_alloc_token(sizeof(int)); }

// Allocations outside of any function use an empty function name.
// CHECK-LABEL: define internal void @__cxx_global_var_init(
// CHECK: call {{.*}} @_Znwm(i64 noundef 4){{.*}}, !alloc_token [[META_GLOBAL:![0-9]+]]
void *global = new int;

// CHECK-DAG: [[META_FUNC]] = !{!"int", i1 false, !"test_func"}
// CHECK-DAG: [[META_UNKNOWN]] = !{!"", i1 false, !"test_func"}
// CHECK-DAG: [[META_METHOD]] = !{!"WithPtr", i1 true, !"ns::S::method"}
// CHECK-DAG: [[META_TMPL]] = !{!"long", i1 false, !"Tmpl<long>::alloc"}
// CHECK-DAG: [[META_LAMBDA]] = !{!"int", i1 false, !"test_lambda()::(lambda)::operator()"}
// CHECK-DAG: [[META_BUILTIN]] = !{!"int", i1 false, !"test_builtin"}
// CHECK-DAG: [[META_GLOBAL]] = !{!"int", i1 false, !""}
