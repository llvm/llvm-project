// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s -fsanitize=function -fno-sanitize-recover=all | FileCheck %s --check-prefixes=CHECK,64
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s -fsanitize=function -fno-sanitize-recover=all | FileCheck %s --check-prefixes=CHECK,64
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -emit-llvm -o - %s -fsanitize=function -fno-sanitize-recover=all | FileCheck %s --check-prefixes=CHECK,64
// RUN: %clang_cc1 -triple arm-none-eabi -emit-llvm -o - %s -fsanitize=function -fno-sanitize-recover=all | FileCheck %s --check-prefixes=CHECK,ARM,32

// CHECK: define{{.*}} void @_Z3funv() #0 !func_sanitize ![[FUNCSAN:.*]] {
void fun() {}

// CHECK-LABEL: define{{.*}} void @_Z6callerPFvvE(ptr noundef %f)
// ARM:   ptrtoint ptr {{.*}} to i32, !nosanitize !5
// ARM:   and i32 {{.*}}, -2, !nosanitize !5
// ARM:   inttoptr i32 {{.*}} to ptr, !nosanitize !5
// CHECK: getelementptr <{ i32, i32 }>, ptr {{.*}}, i32 -1, i32 0, !nosanitize
// CHECK: load i32, ptr {{.*}}, align {{.*}}, !nosanitize
// CHECK: icmp eq i32 {{.*}}, -1056584962, !nosanitize
// CHECK: br i1 {{.*}}, label %[[LABEL1:.*]], label %[[LABEL4:.*]], !nosanitize
// CHECK: [[LABEL1]]:
// CHECK: getelementptr <{ i32, i32 }>, ptr {{.*}}, i32 -1, i32 1, !nosanitize
// CHECK: load i32, ptr {{.*}}, align {{.*}}, !nosanitize
// CHECK: icmp eq i32 {{.*}}, 905068220, !nosanitize
// CHECK: br i1 {{.*}}, label %[[LABEL3:.*]], label %[[LABEL2:[^,]*]], {{.*}}!nosanitize
// CHECK: [[LABEL2]]:
// 64:    call void @__ubsan_handle_function_type_mismatch_abort(ptr @[[#]], i64 %[[#]]) #[[#]], !nosanitize
// 32:    call void @__ubsan_handle_function_type_mismatch_abort(ptr @[[#]], i32 %[[#]]) #[[#]], !nosanitize
// CHECK-NEXT: unreachable, !nosanitize
// CHECK-EMPTY:
// CHECK-NEXT: [[LABEL3]]:
// CHECK: br label %[[LABEL4]], !nosanitize
// CHECK-EMPTY:
// CHECK-NEXT: [[LABEL4]]:
// CHECK-NEXT:   call void
// CHECK-NEXT:   ret void
void caller(void (*f)()) { f(); }

// CHECK: ![[FUNCSAN]] = !{i32 -1056584962, i32 905068220}
