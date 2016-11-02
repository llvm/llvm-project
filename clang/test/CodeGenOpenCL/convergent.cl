// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm %s -o - | opt -instnamer -S | FileCheck %s

void convfun(void) __attribute__((convergent));
void non_convfun(void);
void nodupfun(void) __attribute__((noduplicate));

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
// CHECK: define spir_func void @test_merge_if(i32 %[[a:.+]])
// CHECK: %[[tobool:.+]] = icmp eq i32 %[[a]], 0
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
// CHECK-LABEL: ret void

void test_merge_if(int a) {
  if (a) {
    f();
  }
  non_convfun();
  if (a) {
    g();
  }
}

// CHECK-DAG: declare spir_func void @f()
// CHECK-DAG: declare spir_func void @non_convfun()
// CHECK-DAG: declare spir_func void @g()

// Test two if's are not merged.
// CHECK: define spir_func void @test_no_merge_if(i32 %[[a:.+]])
// CHECK:  %[[tobool:.+]] = icmp eq i32 %[[a]], 0
// CHECK: br i1 %[[tobool]], label %[[if_end:.+]], label %[[if_then:.+]]
// CHECK: [[if_then]]:
// CHECK: tail call spir_func void @f()
// CHECK-NOT: call spir_func void @convfun()
// CHECK-NOT: call spir_func void @g()
// CHECK: br label %[[if_end]]
// CHECK: [[if_end]]:
// CHECK:  %[[tobool_pr:.+]] = phi i1 [ true, %[[if_then]] ], [ false, %{{.+}} ]
// CHECK:  tail call spir_func void @convfun() #[[attr5:.+]]
// CHECK:  br i1 %[[tobool_pr]], label %[[if_then2:.+]], label %[[if_end3:.+]]
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

// CHECK: declare spir_func void @convfun(){{[^#]*}} #[[attr2:[0-9]+]]

// Test loop is unrolled for convergent function.
// CHECK-LABEL: define spir_func void @test_unroll()
// CHECK:  tail call spir_func void @convfun() #[[attr5:[0-9]+]]
// CHECK:  tail call spir_func void @convfun() #[[attr5]]
// CHECK:  tail call spir_func void @convfun() #[[attr5]]
// CHECK:  tail call spir_func void @convfun() #[[attr5]]
// CHECK:  tail call spir_func void @convfun() #[[attr5]]
// CHECK:  tail call spir_func void @convfun() #[[attr5]]
// CHECK:  tail call spir_func void @convfun() #[[attr5]]
// CHECK:  tail call spir_func void @convfun() #[[attr5]]
// CHECK:  tail call spir_func void @convfun() #[[attr5]]
// CHECK:  tail call spir_func void @convfun() #[[attr5]]
// CHECK-LABEL:  ret void

void test_unroll() {
  for (int i = 0; i < 10; i++)
    convfun();
}

// Test loop is not unrolled for noduplicate function.
// CHECK-LABEL: define spir_func void @test_not_unroll()
// CHECK:  br label %[[for_body:.+]]
// CHECK: [[for_cond_cleanup:.+]]:
// CHECK:  ret void
// CHECK: [[for_body]]:
// CHECK:  tail call spir_func void @nodupfun() #[[attr6:[0-9]+]]
// CHECK-NOT: call spir_func void @nodupfun()
// CHECK:  br i1 %{{.+}}, label %[[for_body]], label %[[for_cond_cleanup]]

void test_not_unroll() {
  for (int i = 0; i < 10; i++)
    nodupfun();
}

// CHECK: declare spir_func void @nodupfun(){{[^#]*}} #[[attr3:[0-9]+]]

// CHECK-DAG: attributes #[[attr2]] = { {{[^}]*}}convergent{{[^}]*}} }
// CHECK-DAG: attributes #[[attr3]] = { {{[^}]*}}noduplicate{{[^}]*}} }
// CHECK-DAG: attributes #[[attr5]] = { {{[^}]*}}convergent{{[^}]*}} }
// CHECK-DAG: attributes #[[attr6]] = { {{[^}]*}}noduplicate{{[^}]*}} }
