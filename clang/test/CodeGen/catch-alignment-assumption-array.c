// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -fsanitize=alignment -fno-sanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-trap=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER: @[[CHAR:.*]] = {{.*}} c"'char *'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[ALIGNMENT_ASSUMPTION:.*]] = {{.*}}, i32 30, i32 35 }, {{.*}} @[[CHAR]] }

void *caller(void) {
  char str[] = "";
  // CHECK:                           define{{.*}}
  // CHECK-NEXT:                      entry:
  // CHECK-NEXT:                        %[[STR:.*]] = alloca [1 x i8], align 1
  // CHECK-NEXT:                        call void @llvm.memset.p0.i64(ptr align 1 %[[STR]], i8 0, i64 1, i1 false)
  // CHECK-NEXT:                        %[[ARRAYDECAY:.*]] = getelementptr inbounds [1 x i8], ptr %[[STR]], i64 0, i64 0
  // CHECK-SANITIZE-NEXT:               %[[PTRINT:.*]] = ptrtoint ptr %[[ARRAYDECAY]] to i64
  // CHECK-SANITIZE-NEXT:               %[[MASKEDPTR:.*]] = and i64 %[[PTRINT]], 0
  // CHECK-SANITIZE-NEXT:               %[[MASKCOND:.*]] = icmp eq i64 %[[MASKEDPTR]], 0
  // CHECK-SANITIZE-NEXT:               %[[PTRINT_DUP:.*]] = ptrtoint ptr %[[ARRAYDECAY]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[MASKCOND]], label %[[CONT:.*]], label %[[HANDLER_ALIGNMENT_ASSUMPTION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_ALIGNMENT_ASSUMPTION]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_alignment_assumption_abort(ptr @[[ALIGNMENT_ASSUMPTION]], i64 %[[PTRINT_DUP]], i64 1, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_alignment_assumption(ptr @[[ALIGNMENT_ASSUMPTION]], i64 %[[PTRINT_DUP]], i64 1, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 23){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        call void @llvm.assume(i1 true) [ "align"(ptr %[[ARRAYDECAY]], i64 1) ] 
  // CHECK-NEXT:                        ret ptr %[[ARRAYDECAY]]
  // CHECK-NEXT:                      }
  return __builtin_assume_aligned(str, 1);
}
