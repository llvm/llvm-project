// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s -fsanitize=function -fno-sanitize-recover=all | FileCheck %s

// CHECK: @[[PROXY:.*]] = private unnamed_addr constant ptr @_ZTIFvvE
// CHECK: define{{.*}} void @_Z3funv() #0 !func_sanitize ![[FUNCSAN:.*]] {
void fun() {}

// CHECK-LABEL: define{{.*}} void @_Z6callerPFvvE(ptr noundef %f)
// CHECK: getelementptr <{ i32, i32 }>, ptr {{.*}}, i32 0, i32 0, !nosanitize
// CHECK: load i32, ptr {{.*}}, align {{.*}}, !nosanitize
// CHECK: icmp eq i32 {{.*}}, 846595819, !nosanitize
// CHECK: br i1 {{.*}}, label %[[LABEL1:.*]], label %[[LABEL4:.*]], !nosanitize
// CHECK: [[LABEL1]]:
// CHECK: getelementptr <{ i32, i32 }>, ptr {{.*}}, i32 0, i32 1, !nosanitize
// CHECK: load i32, ptr {{.*}}, align {{.*}}, !nosanitize
// CHECK: icmp eq ptr {{.*}}, @_ZTIFvvE, !nosanitize
// CHECK: br i1 {{.*}}, label %[[LABEL3:.*]], label %[[LABEL2:[^,]*]], {{.*}}!nosanitize
// CHECK: [[LABEL2]]:
// CHECK: call void @__ubsan_handle_function_type_mismatch_v1_abort(ptr {{.*}}, i64 {{.*}}, i64 {{.*}}, i64 {{.*}}) #{{.*}}, !nosanitize
// CHECK-NOT: unreachable
// CHECK: br label %[[LABEL3]], !nosanitize
// CHECK: [[LABEL3]]:
// CHECK: br label %[[LABEL4]], !nosanitize
void caller(void (*f)()) { f(); }

// CHECK: ![[FUNCSAN]] = !{i32 846595819, ptr @[[PROXY]]}
