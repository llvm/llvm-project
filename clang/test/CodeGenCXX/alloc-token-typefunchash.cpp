// Test that the typefunchash modes add the name of the function containing the
// allocation to the !alloc_token metadata, and always emit metadata.
//
// RUN: %clang_cc1 -fsanitize=alloc-token -falloc-token-mode=typefunchash -triple x86_64-linux-gnu -std=c++20 -fblocks -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s
// RUN: %clang_cc1 -fsanitize=alloc-token -falloc-token-mode=typefunchashpointersplit -triple x86_64-linux-gnu -std=c++20 -fblocks -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

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

template <typename T>
struct Holder {
  static void *alloc() { return new int; }
};

// Unnamed types are printed without their source location.
// CHECK-LABEL: define {{.*}} @"_ZN6HolderIZ17test_unnamed_typevE{{.*}}E5allocEv"(
// CHECK: call {{.*}} @_Znwm(i64 noundef 4){{.*}}, !alloc_token [[META_UNNAMED:![0-9]+]]
void test_unnamed_type() {
  auto L = [] {};
  sink = Holder<decltype(L)>::alloc();
}

// Inline namespaces are always omitted, regardless of other declarations such
// as ns2::inl(int).
namespace ns2 {
void *inl(int);
inline namespace v1 {
// CHECK-LABEL: define {{.*}} @_ZN3ns22v13inlEv(
// CHECK: call {{.*}} @_Znwm(i64 noundef 4){{.*}}, !alloc_token [[META_INLINE_NS:![0-9]+]]
void *inl() { return new int; }
} // namespace v1
} // namespace ns2

// Lambdas, blocks, and captured statements use the enclosing function.
// CHECK-LABEL: define {{.*}} @"_ZZ11test_lambdavENK{{.*}}clEv"(
// CHECK: call {{.*}} @_Znwm(i64 noundef 4){{.*}}, !alloc_token [[META_LAMBDA:![0-9]+]]
void test_lambda() {
  auto L = [] { return new int; };
  sink = L();
}

// CHECK-LABEL: define {{.*}} @___Z13test_closuresv_block_invoke(
// CHECK: call {{.*}} @_Znwm(i64 noundef 4){{.*}}, !alloc_token [[META_CLOSURES:![0-9]+]]
// CHECK-LABEL: define {{.*}} @__captured_stmt(
// CHECK: call {{.*}} @_Znwm(i64 noundef 4){{.*}}, !alloc_token [[META_CLOSURES]]
void test_closures() {
  void (^B)(void) = ^{ sink = new int; };
  B();
#pragma clang __debug captured
  { sink = new int; }
}

// Runtime use of the builtin, which is not a constant expression in this mode.
// CHECK-LABEL: define {{.*}} @_Z12test_builtinv(
// CHECK: call i64 @llvm.alloc.token.id.i64(metadata [[META_BUILTIN:![0-9]+]])
unsigned long test_builtin() { return __builtin_infer_alloc_token(sizeof(int)); }

// Allocations outside of any function use an empty function name, also in
// lambdas.
// CHECK-LABEL: define internal void @__cxx_global_var_init(
// CHECK: call {{.*}} @_Znwm(i64 noundef 4){{.*}}, !alloc_token [[META_GLOBAL:![0-9]+]]
void *global = new int;
// CHECK-LABEL: define {{.*}} @"_ZNK{{.*}}clEv"(
// CHECK: call {{.*}} @_Znwm(i64 noundef 4){{.*}}, !alloc_token [[META_GLOBAL]]
void *global_lambda = [] { return new int; }();

// Closures nested in lambdas also use the enclosing function.
// CHECK-LABEL: define {{.*}} @"___ZZ11test_nestedvENK{{.*}}clEv_block_invoke"(
// CHECK: call {{.*}} @_Znwm(i64 noundef 4){{.*}}, !alloc_token [[META_NESTED:![0-9]+]]
// CHECK-LABEL: define {{.*}} @__captured_stmt{{.*}}(
// CHECK: call {{.*}} @_Znwm(i64 noundef 4){{.*}}, !alloc_token [[META_NESTED]]
void test_nested() {
  auto L = [] {
    void (^B)(void) = ^{ sink = new int; };
    B();
#pragma clang __debug captured
    { sink = new int; }
  };
  L();
}

// CHECK-DAG: [[META_FUNC]] = !{!"int", i1 false, !"test_func"}
// CHECK-DAG: [[META_UNKNOWN]] = !{!"", i1 false, !"test_func"}
// CHECK-DAG: [[META_METHOD]] = !{!"WithPtr", i1 true, !"ns::S::method"}
// CHECK-DAG: [[META_TMPL]] = !{!"long", i1 false, !"Tmpl<long>::alloc"}
// CHECK-DAG: [[META_UNNAMED]] = !{!"int", i1 false, !"Holder<(lambda)>::alloc"}
// CHECK-DAG: [[META_INLINE_NS]] = !{!"int", i1 false, !"ns2::inl"}
// CHECK-DAG: [[META_LAMBDA]] = !{!"int", i1 false, !"test_lambda"}
// CHECK-DAG: [[META_CLOSURES]] = !{!"int", i1 false, !"test_closures"}
// CHECK-DAG: [[META_BUILTIN]] = !{!"int", i1 false, !"test_builtin"}
// CHECK-DAG: [[META_GLOBAL]] = !{!"int", i1 false, !""}
// CHECK-DAG: [[META_NESTED]] = !{!"int", i1 false, !"test_nested"}
