// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --input-file=%t-og.ll --check-prefix=OGCG %s

// Test that we handle C++ cast expressions as lvalues correctly.
// This used to assert with "Use emitCastLValue below, remove me when adding testcase"
// at CIRGenExpr.cpp:2720

// Test reinterpret_cast as lvalue
void test_reinterpret_cast_lvalue() {
  int x = 42;
  reinterpret_cast<char&>(x) = 'A';
}

// CIR-LABEL: cir.func{{.*}}@_Z28test_reinterpret_cast_lvaluev()
// CIR: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR: cir.cast bitcast{{.*}}!cir.ptr<!s32i>{{.*}}!cir.ptr<!s8i>
// CIR: cir.store{{.*}}!s8i, !cir.ptr<!s8i>

// LLVM-LABEL: define{{.*}}@_Z28test_reinterpret_cast_lvaluev()
// LLVM: alloca i32
// LLVM: store i8 65, ptr

// OGCG-LABEL: define{{.*}}@_Z28test_reinterpret_cast_lvaluev()
// OGCG: alloca i32
// OGCG: store i8 65, ptr

// Test const_cast as lvalue
void test_const_cast_lvalue() {
  const int x = 0;
  const_cast<int&>(x) = 1;
}

// CIR-LABEL: cir.func{{.*}}@_Z22test_const_cast_lvaluev()
// CIR: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR: cir.store{{.*}}!s32i, !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}}@_Z22test_const_cast_lvaluev()
// LLVM: alloca i32
// LLVM: store i32 1, ptr

// OGCG-LABEL: define{{.*}}@_Z22test_const_cast_lvaluev()
// OGCG: alloca i32
// OGCG: store i32 1, ptr

// Test const_cast in template context (from crashes/const-cast-lvalue.cpp)
int global_a;
struct S {
  using type = int;
  static type foo() { return const_cast<int &>(global_a); }
};
template <typename> struct T {
  static bool bar() { return S::foo(); }
};
template <typename... Args, typename H> void baz(H) { (T<Args>::bar() || ...); }
class C {
  int member;
public:
  void method() { baz<int>(member); }
};
void caller() {
  C obj;
  obj.method();
}

// CIR-LABEL: cir.func{{.*}}@_ZN1S3fooEv()
// CIR: cir.get_global @global_a

// LLVM-LABEL: define{{.*}}@_ZN1S3fooEv()
// LLVM: load{{.*}}@global_a

// OGCG-LABEL: define{{.*}}@_ZN1S3fooEv()
// OGCG: load{{.*}}@global_a
