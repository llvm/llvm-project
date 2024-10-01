// RUN: %clang_cc1 -fclangir -triple spirv64-unknown-unknown -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -fclangir -triple spirv64-unknown-unknown -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM

// In ClangIR for OpenCL, all functions should be marked convergent.
// In LLVM IR, it is initially assumed convergent, but can be deduced to not require it.

// CIR:      #fn_attr[[CONV_NOINLINE_ATTR:[0-9]*]] = #cir<extra({convergent = #cir.convergent, inline = #cir.inline<no>
// CIR-NEXT: #fn_attr[[CONV_DECL_ATTR:[0-9]*]] = #cir<extra({convergent = #cir.convergent

__attribute__((noinline))
void non_convfun(void) {
  volatile int* p;
  *p = 0;
}
// CIR: cir.func @non_convfun(){{.*}} extra(#fn_attr[[CONV_NOINLINE_ATTR]])
// LLVM: define{{.*}} spir_func void @non_convfun() local_unnamed_addr #[[NON_CONV_ATTR:[0-9]+]]
// LLVM: ret void

// External functions should be assumed convergent.
void f(void);
// CIR: cir.func{{.+}} @f(){{.*}} extra(#fn_attr[[CONV_DECL_ATTR]])
// LLVM: declare {{.+}} spir_func void @f() local_unnamed_addr #[[CONV_ATTR:[0-9]+]]
void g(void);
// CIR: cir.func{{.+}} @g(){{.*}} extra(#fn_attr[[CONV_DECL_ATTR]])
// LLVM: declare {{.+}} spir_func void @g() local_unnamed_addr #[[CONV_ATTR]]

// Test two if's are merged and non_convfun duplicated.
void test_merge_if(int a) {
  if (a) {
    f();
  }
  non_convfun();
  if (a) {
    g();
  }
}
// CIR: cir.func @test_merge_if{{.*}} extra(#fn_attr[[CONV_DECL_ATTR]])

// The LLVM IR below is equivalent to:
//    if (a) {
//      f();
//      non_convfun();
//      g();
//    } else {
//      non_convfun();
//    }

// LLVM-LABEL: define{{.*}} spir_func void @test_merge_if
// LLVM:         %[[tobool:.+]] = icmp eq i32 %[[ARG:.+]], 0
// LLVM:         br i1 %[[tobool]], label %[[if_end3_critedge:[^,]+]], label %[[if_then:[^,]+]]

// LLVM:       [[if_end3_critedge]]:
// LLVM:         tail call spir_func void @non_convfun()
// LLVM:         br label %[[if_end3:[^,]+]]

// LLVM:       [[if_then]]:
// LLVM:         tail call spir_func void @f()
// LLVM:         tail call spir_func void @non_convfun()
// LLVM:         tail call spir_func void @g()

// LLVM:         br label %[[if_end3]]

// LLVM:       [[if_end3]]:
// LLVM:         ret void


void convfun(void) __attribute__((convergent));
// CIR: cir.func{{.+}} @convfun(){{.*}} extra(#fn_attr[[CONV_DECL_ATTR]])
// LLVM: declare {{.+}} spir_func void @convfun() local_unnamed_addr #[[CONV_ATTR]]

// Test two if's are not merged.
void test_no_merge_if(int a) {
  if (a) {
    f();
  }
  convfun();
  if(a) {
    g();
  }
}
// CIR: cir.func @test_no_merge_if{{.*}} extra(#fn_attr[[CONV_DECL_ATTR]])

// LLVM-LABEL: define{{.*}} spir_func void @test_no_merge_if
// LLVM:         %[[tobool:.+]] = icmp eq i32 %[[ARG:.+]], 0
// LLVM:         br i1 %[[tobool]], label %[[if_end:[^,]+]], label %[[if_then:[^,]+]]
// LLVM:       [[if_then]]:
// LLVM:         tail call spir_func void @f()
// LLVM-NOT:     call spir_func void @convfun()
// LLVM-NOT:     call spir_func void @g()
// LLVM:         br label %[[if_end]]
// LLVM:       [[if_end]]:
// LLVM-NOT:     phi i1
// LLVM:         tail call spir_func void @convfun()
// LLVM:         br i1 %[[tobool]], label %[[if_end3:[^,]+]], label %[[if_then2:[^,]+]]
// LLVM:       [[if_then2]]:
// LLVM:         tail call spir_func void @g()
// LLVM:         br label %[[if_end3:[^,]+]]
// LLVM:       [[if_end3]]:
// LLVM:         ret void


// LLVM attribute definitions.
// LLVM-NOT: attributes #[[NON_CONV_ATTR]] = { {{.*}}convergent{{.*}} }
// LLVM:     attributes #[[CONV_ATTR]] = { {{.*}}convergent{{.*}} }
