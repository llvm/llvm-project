// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s -fsanitize=function -fno-sanitize-recover=all | FileCheck %s --check-prefixes=CHECK,GNU,64
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -emit-llvm -o - %s -fsanitize=function -fno-sanitize-recover=all | FileCheck %s --check-prefixes=CHECK,MSVC,64
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s -fsanitize=function -fno-sanitize-recover=all | FileCheck %s --check-prefixes=CHECK,GNU,64
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -emit-llvm -o - %s -fsanitize=function -fno-sanitize-recover=all | FileCheck %s --check-prefixes=CHECK,GNU,64
// RUN: %clang_cc1 -triple arm-none-eabi -emit-llvm -o - %s -fsanitize=function -fno-sanitize-recover=all | FileCheck %s --check-prefixes=CHECK,ARM,GNU,32

// GNU:  define{{.*}} void @_Z3funv() #0 !func_sanitize ![[FUNCSAN:.*]] {
// MSVC: define{{.*}} void @"?fun@@YA?A?<auto>@@XZ"() #0 !func_sanitize ![[FUNCSAN:.*]] {
auto fun() {}

// GNU-LABEL:  define{{.*}} void @_Z6callerv()
// MSVC-LABEL: define{{.*}} void @"?caller@@YAXXZ"()
// ARM:   ptrtoint ptr {{.*}} to i32, !nosanitize !4
// ARM:   and i32 {{.*}}, -2, !nosanitize !4
// ARM:   inttoptr i32 {{.*}} to ptr, !nosanitize !4
// CHECK: getelementptr <{ i32, i32 }>, ptr {{.*}}, i32 -1, i32 0, !nosanitize
// CHECK: load i32, ptr {{.*}}, align {{.*}}, !nosanitize
// CHECK: icmp eq i32 {{.*}}, -1056584962, !nosanitize
// CHECK: br i1 {{.*}}, label %[[LABEL1:.*]], label %[[LABEL4:.*]], !nosanitize
// CHECK: [[LABEL1]]:
// CHECK: getelementptr <{ i32, i32 }>, ptr {{.*}}, i32 -1, i32 1, !nosanitize
// CHECK: load i32, ptr {{.*}}, align {{.*}}, !nosanitize
// GNU:   icmp eq i32 {{.*}}, 905068220, !nosanitize
// MSVC:  icmp eq i32 {{.*}}, -1600339357, !nosanitize
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
void caller() {
  auto a = fun;
  a();
}

// GNU:  ![[FUNCSAN]] = !{i32 -1056584962, i32 905068220}
// MSVC: ![[FUNCSAN]] = !{i32 -1056584962, i32 -1600339357}
