// RUN: %clang_cc1 -triple aarch64-linux-android29 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-linux-android29 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-linux-android29 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Verify that __attribute__((annotate(...))) on C++ entities works through
// the CXX ABI lowering pipeline. The CXXABILowering pass walks all attributes
// on functions/globals and asserts they are "legal" (i.e. don't need ABI
// conversion). cir::AnnotationAttr was added to that legal-list; this test
// covers that path by exercising annotated members and free functions in a
// translation unit that goes through C++ codegen (mangling, member functions,
// constructor emission).

// All globals are emitted before any function in CIR/LLVM/OGCG output;
// collect global-section checks up here so subsequent function CHECKs can
// rely on strict-order matching for the function section.
// CIR-DAG: cir.global external @{{.*ns.*g.*}} = #cir.int<5> : !s32i [#cir.annotation<"ns_global_ann">]
// LLVM-DAG: @{{.*ns.*g.*}} = global i32 5
// OGCG-DAG: @{{.*ns.*g.*}} = {{.*}}global i32 5
// OGCG-DAG: @llvm.global.annotations = appending global

// CIR-side LLVM lowering also emits @llvm.global.annotations.
// LLVM-DAG: @llvm.global.annotations = appending global

struct __attribute__((annotate("type_ann"))) Tagged {
  int x;
  __attribute__((annotate("method_ann")))
  int get() { return x; }

  __attribute__((annotate("static_method_ann")))
  static int sget() { return 7; }
};

// CIR: cir.func {{.*}} @{{.*Tagged.*get.*}}({{.*}}) {{.*}}[#cir.annotation<"method_ann">]
// CIR: cir.func {{.*}} @{{.*Tagged.*sget.*}}() {{.*}}[#cir.annotation<"static_method_ann">]

// LLVM: define{{.*}} i32 @{{.*Tagged.*get.*}}
// LLVM: define{{.*}} i32 @{{.*Tagged.*sget.*}}

// OGCG-DAG: define{{.*}} i32 @{{.*Tagged.*get.*}}
// OGCG-DAG: define{{.*}} i32 @{{.*Tagged.*sget.*}}

__attribute__((annotate("free_fn_ann")))
int use_tagged(Tagged *t) {
  return t->get() + Tagged::sget();
}

// CIR: cir.func {{.*}} @{{.*use_tagged.*}}({{.*}}) {{.*}}[#cir.annotation<"free_fn_ann">]
// LLVM: define{{.*}} i32 @{{.*use_tagged.*}}
// OGCG-DAG: define{{.*}} i32 @{{.*use_tagged.*}}

// Annotated namespace-scope global (C++ name mangling exercises a different
// path than C globals). The CIR-DAG / LLVM-DAG / OGCG-DAG checks for the
// emitted global live in the globals block at the top.
namespace ns {
__attribute__((annotate("ns_global_ann")))
int g = 5;
}
