// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -fsanitize=alignment -fno-sanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-trap=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER: @[[CHAR:.*]] = {{.*}} c"'B *'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_100_ALIGNMENT_ASSUMPTION:.*]] = {{.*}}, i32 100, i32 35 }, {{.*}} @[[CHAR]] }

struct A { int n; };
struct B { int n; };
struct C : A, B {};

void *f(C *c) {
  // CHECK:                             define {{.*}} ptr @{{.*}}(ptr noundef %[[C:.*]]) {{.*}} {
  // CHECK-NEXT:                        [[ENTRY:.*]]:
  // CHECK-NEXT:                          %[[C_ADDR:.*]] = alloca ptr
  // CHECK-NEXT:                          store ptr %[[C]], ptr %[[C_ADDR]]
  // CHECK-NEXT:                          %[[C_RELOAD:.*]] = load ptr, ptr %[[C_ADDR]]
  // CHECK-NEXT:                          %[[IS_NULL:.*]] = icmp eq ptr %[[C_RELOAD]], null
  // CHECK-NEXT:                          br i1 %[[IS_NULL]], label %[[CAST_END:[^,]+]], label %[[CAST_NOT_NULL:[^,]+]]
  // CHECK:                             [[CAST_NOT_NULL]]:
  // CHECK-NOSANITIZE-NEXT:               %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[C_RELOAD]], i64 4
  // CHECK-NOSANITIZE-NEXT:               br label %[[CAST_END]]
  // CHECK-SANITIZE-NEXT:                 %[[PTRTOINT:.*]] = ptrtoint ptr %[[C_RELOAD]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:                 %[[MASKEDPTR:.*]] = and i64 %[[PTRTOINT]], 3, !nosanitize
  // CHECK-SANITIZE-NEXT:                 %[[MASKCOND:.*]] = icmp eq i64 %[[MASKEDPTR]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:                 br i1 %[[MASKCOND]], label %[[CONT:[^,]+]], label %[[HANDLER_TYPE_MISMATCH:[^,]+]]
  // CHECK-SANITIZE:                    [[HANDLER_TYPE_MISMATCH]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:       call void @__ubsan_handle_type_mismatch_v1_abort(
  // CHECK-SANITIZE-RECOVER-NEXT:         call void @__ubsan_handle_type_mismatch_v1(
  // CHECK-SANITIZE-TRAP-NEXT:            call void @llvm.ubsantrap(
  // CHECK-SANITIZE-UNREACHABLE-NEXT:     unreachable, !nosanitize
  // CHECK-SANITIZE:                    [[CONT]]:
  // CHECK-SANITIZE-NEXT:                 %[[ADD_PTR:.*]] = getelementptr inbounds i8, ptr %[[C_RELOAD]], i64 4
  // CHECK-SANITIZE-NEXT:                 br label %[[CAST_END]]
  // CHECK:                             [[CAST_END]]:
  // CHECK-NOSANITIZE-NEXT:               %[[CAST_RESULT:.*]] = phi ptr [ %[[ADD_PTR]], %[[CAST_NOT_NULL]] ], [ null, %[[ENTRY]] ]
  // CHECK-NOSANITIZE-NEXT:               call void @llvm.assume(i1 true) [ "align"(ptr %[[CAST_RESULT]], i64 8) ]
  // CHECK-NOSANITIZE-NEXT:               ret ptr %[[CAST_RESULT]]
  // CHECK-NOSANITIZE-NEXT:              }
  // CHECK-SANITIZE-NEXT:                 %[[CAST_RESULT:.*]] = phi ptr [ %[[ADD_PTR]], %[[CONT]] ], [ null, %[[ENTRY]] ]
  // CHECK-SANITIZE-NEXT:                 %[[PTRINT:.*]] = ptrtoint ptr %[[CAST_RESULT]] to i64
  // CHECK-SANITIZE-NEXT:                 %[[MASKEDPTR:.*]] = and i64 %[[PTRINT]], 7
  // CHECK-SANITIZE-NEXT:                 %[[MASKCOND:.*]] = icmp eq i64 %[[MASKEDPTR]], 0
  // CHECK-SANITIZE-NEXT:                 %[[PTRINT_DUP:.*]] = ptrtoint ptr %[[CAST_RESULT]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:                 br i1 %[[MASKCOND]], label %[[CONT1:.*]], label %[[HANDLER_ALIGNMENT_ASSUMPTION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                    [[HANDLER_ALIGNMENT_ASSUMPTION]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:       call void @__ubsan_handle_alignment_assumption_abort(ptr @[[LINE_100_ALIGNMENT_ASSUMPTION]], i64 %[[PTRINT_DUP]], i64 8, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT:         call void @__ubsan_handle_alignment_assumption(ptr @[[LINE_100_ALIGNMENT_ASSUMPTION]], i64 %[[PTRINT_DUP]], i64 8, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT:            call void @llvm.ubsantrap(i8 23){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:     unreachable, !nosanitize
  // CHECK-SANITIZE:                    [[CONT1]]:
  // CHECK-SANITIZE-NEXT:                 call void @llvm.assume(i1 true) [ "align"(ptr %[[CAST_RESULT]], i64 8) ]
  // CHECK-SANITIZE-NEXT:                 ret ptr %[[CAST_RESULT]]
  // CHECK-SANITIZE-NEXT:                }
#line 100
  return __builtin_assume_aligned((B*)c, 8);
}
