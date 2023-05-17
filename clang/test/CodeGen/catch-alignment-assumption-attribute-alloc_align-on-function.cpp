// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-NOSANITIZE
// RUN: %clang_cc1 -fsanitize=alignment -fno-sanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-trap=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER: @[[CHAR:.*]] = {{.*}} c"'char **'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_100_ALIGNMENT_ASSUMPTION:.*]] = {{.*}}, i32 100, i32 10 }, {{.*}} @[[CHAR]] }

char **__attribute__((alloc_align(2)))
passthrough(char **x, unsigned long alignment) {
  // CHECK:      define{{.*}} ptr @[[PASSTHROUGH:.*]](ptr noundef %[[X:.*]], i64 noundef %[[ALIGNMENT:.*]])
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %[[X_ADDR:.*]] = alloca ptr, align 8
  // CHECK-NEXT:   %[[ALIGNMENT_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:   store ptr %[[X]], ptr %[[X_ADDR]], align 8
  // CHECK-NEXT:   store i64 %[[ALIGNMENT]], ptr %[[ALIGNMENT_ADDR]], align 8
  // CHECK-NEXT:   %[[X_RELOADED:.*]] = load ptr, ptr %[[X_ADDR]], align 8
  // CHECK-NEXT:   ret ptr %[[X_RELOADED]]
  // CHECK-NEXT: }
  return x;
}

char **caller(char **x) {
  // CHECK:                           define{{.*}} ptr @{{.*}}(ptr noundef %[[X:.*]])
  // CHECK-NEXT:                      entry:
  // CHECK-NEXT:                        %[[X_ADDR:.*]] = alloca ptr, align 8
  // CHECK-NEXT:                        store ptr %[[X]], ptr %[[X_ADDR]], align 8
  // CHECK-NEXT:                        %[[X_RELOADED:.*]] = load ptr, ptr %[[X_ADDR]], align 8
  // CHECK-NOSANITIZE-NEXT:             %[[X_RETURNED:.*]] = call noundef align 128 ptr @[[PASSTHROUGH]](ptr noundef %[[X_RELOADED]], i64 noundef 128)
  // CHECK-SANITIZE-NEXT:               %[[X_RETURNED:.*]] = call noundef ptr @[[PASSTHROUGH]](ptr noundef %[[X_RELOADED]], i64 noundef 128)
  // CHECK-SANITIZE-NEXT:               %[[PTRINT:.*]] = ptrtoint ptr %[[X_RETURNED]] to i64
  // CHECK-SANITIZE-NEXT:               %[[MASKEDPTR:.*]] = and i64 %[[PTRINT]], 127
  // CHECK-SANITIZE-NEXT:               %[[MASKCOND:.*]] = icmp eq i64 %[[MASKEDPTR]], 0
  // CHECK-SANITIZE-NEXT:               %[[PTRINT_DUP:.*]] = ptrtoint ptr %[[X_RETURNED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[MASKCOND]], label %[[CONT:.*]], label %[[HANDLER_ALIGNMENT_ASSUMPTION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_ALIGNMENT_ASSUMPTION]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_alignment_assumption_abort(ptr @[[LINE_100_ALIGNMENT_ASSUMPTION]], i64 %[[PTRINT_DUP]], i64 128, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_alignment_assumption(ptr @[[LINE_100_ALIGNMENT_ASSUMPTION]], i64 %[[PTRINT_DUP]], i64 128, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 23){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-SANITIZE-NEXT:               call void @llvm.assume(i1 true) [ "align"(ptr %[[X_RETURNED]], i64 128) ]
  // CHECK-NEXT:                        ret ptr %[[X_RETURNED]]
  // CHECK-NEXT:                      }
#line 100
  return passthrough(x, 128);
}
