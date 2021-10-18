// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm %s -o - -fno-experimental-new-pass-manager | opt -instnamer -S | FileCheck -enable-var-scope %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm %s -o - -fexperimental-new-pass-manager | opt -instnamer -S | FileCheck -enable-var-scope %s

// This is initially assumed convergent, but can be deduced to not require it.

// CHECK-LABEL: define{{.*}} spir_func void @non_convfun() local_unnamed_addr #0
// CHECK: ret void
__attribute__((noinline))
void non_convfun(void) {
  volatile int* p;
  *p = 0;
}

void convfun(void) __attribute__((convergent));
void nodupfun(void) __attribute__((noduplicate));

// External functions should be assumed convergent.
void f(void);
void g(void);

// Test two if's are merged and non_convfun duplicated.
// The LLVM IR is equivalent to:
//    if (a) {
//      f();
//      non_convfun();
//      g();
//    } else {
//      non_convfun();
//    }
//
// CHECK-LABEL: define{{.*}} spir_func void @test_merge_if(i32 %a) local_unnamed_addr #1 {
// CHECK: %[[tobool:.+]] = icmp eq i32 %a, 0
// CHECK: br i1 %[[tobool]], label %[[if_end3_critedge:.+]], label %[[if_then:.+]]

// CHECK: [[if_then]]:
// CHECK: tail call spir_func void @f()
// CHECK: tail call spir_func void @non_convfun()
// CHECK: tail call spir_func void @g()

// CHECK: br label %[[if_end3:.+]]

// CHECK: [[if_end3_critedge]]:
// CHECK: tail call spir_func void @non_convfun()
// CHECK: br label %[[if_end3]]

// CHECK: [[if_end3]]:
// CHECK: ret void

void test_merge_if(int a) {
  if (a) {
    f();
  }
  non_convfun();
  if (a) {
    g();
  }
}

// CHECK-DAG: declare spir_func void @f() local_unnamed_addr #2
// CHECK-DAG: declare spir_func void @g() local_unnamed_addr #2


// Test two if's are not merged.
// CHECK-LABEL: define{{.*}} spir_func void @test_no_merge_if(i32 %a) local_unnamed_addr #1
// CHECK:  %[[tobool:.+]] = icmp eq i32 %a, 0
// CHECK: br i1 %[[tobool]], label %[[if_end:.+]], label %[[if_then:.+]]
// CHECK: [[if_then]]:
// CHECK: tail call spir_func void @f()
// CHECK-NOT: call spir_func void @convfun()
// CHECK-NOT: call spir_func void @g()
// CHECK: br label %[[if_end]]
// CHECK: [[if_end]]:
// CHECK-NOT: phi i1
// CHECK:  tail call spir_func void @convfun() #[[attr4:.+]]
// CHECK:  br i1 %[[tobool]], label %[[if_end3:.+]], label %[[if_then2:.+]]
// CHECK: [[if_then2]]:
// CHECK: tail call spir_func void @g()
// CHECK:  br label %[[if_end3:.+]]
// CHECK: [[if_end3]]:
// CHECK-LABEL:  ret void

void test_no_merge_if(int a) {
  if (a) {
    f();
  }
  convfun();
  if(a) {
    g();
  }
}

// CHECK: declare spir_func void @convfun(){{[^#]*}} #2

// Test loop is unrolled for convergent function.
// CHECK-LABEL: define{{.*}} spir_func void @test_unroll() local_unnamed_addr #1
// CHECK:  tail call spir_func void @convfun() #[[attr4:[0-9]+]]
// CHECK:  tail call spir_func void @convfun() #[[attr4]]
// CHECK:  tail call spir_func void @convfun() #[[attr4]]
// CHECK:  tail call spir_func void @convfun() #[[attr4]]
// CHECK:  tail call spir_func void @convfun() #[[attr4]]
// CHECK:  tail call spir_func void @convfun() #[[attr4]]
// CHECK:  tail call spir_func void @convfun() #[[attr4]]
// CHECK:  tail call spir_func void @convfun() #[[attr4]]
// CHECK:  tail call spir_func void @convfun() #[[attr4]]
// CHECK:  tail call spir_func void @convfun() #[[attr4]]
// CHECK-LABEL:  ret void

void test_unroll() {
  for (int i = 0; i < 10; i++)
    convfun();
}

// Test loop is not unrolled for noduplicate function.
// CHECK-LABEL: define{{.*}} spir_func void @test_not_unroll()
// CHECK:  br label %[[for_body:.+]]
// CHECK: [[for_cond_cleanup:.+]]:
// CHECK:  ret void
// CHECK: [[for_body]]:
// CHECK:  tail call spir_func void @nodupfun() #[[attr5:[0-9]+]]
// CHECK-NOT: call spir_func void @nodupfun()
// CHECK:  br i1 %{{.+}}, label %[[for_body]], label %[[for_cond_cleanup]]

void test_not_unroll() {
  for (int i = 0; i < 10; i++)
    nodupfun();
}

// CHECK: declare spir_func void @nodupfun(){{[^#]*}} #[[attr3:[0-9]+]]

// CHECK-LABEL: @assume_convergent_asm
// CHECK: tail call void asm sideeffect "s_barrier", ""() #5
kernel void assume_convergent_asm()
{
  __asm__ volatile("s_barrier");
}

// CHECK: attributes #0 = { nofree noinline norecurse nounwind "
// CHECK: attributes #1 = { {{[^}]*}}convergent{{[^}]*}} }
// CHECK: attributes #2 = { {{[^}]*}}convergent{{[^}]*}} }
// CHECK: attributes #3 = { {{[^}]*}}convergent noduplicate{{[^}]*}} }
// CHECK: attributes #4 = { {{[^}]*}}convergent{{[^}]*}} }
// CHECK: attributes #5 = { {{[^}]*}}convergent{{[^}]*}} }
// CHECK: attributes #6 = { {{[^}]*}}convergent noduplicate{{[^}]*}} }
