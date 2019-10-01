// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-NOSANITIZE
// RUN: %clang_cc1 -fsanitize=alignment -fno-sanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-trap=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER: @[[CHAR:.*]] = {{.*}} c"'char **'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_100_ALIGNMENT_ASSUMPTION:.*]] = {{.*}}, i32 100, i32 10 }, {{.*}}* @[[CHAR]] }

char **passthrough(__attribute__((align_value(0x80000000))) char **x) {
  // CHECK-NOSANITIZE:                define i8** @{{.*}}(i8** align 536870912 %[[X:.*]])
  // CHECK-SANITIZE:                  define i8** @{{.*}}(i8** %[[X:.*]])
  // CHECK-NEXT:                      [[entry:.*]]:
  // CHECK-NEXT:                        %[[X_ADDR:.*]] = alloca i8**, align 8
  // CHECK-NEXT:                        store i8** %[[X]], i8*** %[[X_ADDR]], align 8
  // CHECK-NEXT:                        %[[X_RELOADED:.*]] = load i8**, i8*** %[[X_ADDR]], align 8
  // CHECK-SANITIZE-NEXT:               %[[PTRINT:.*]] = ptrtoint i8** %[[X_RELOADED]] to i64
  // CHECK-SANITIZE-NEXT:               %[[MASKEDPTR:.*]] = and i64 %[[PTRINT]], 2147483647
  // CHECK-SANITIZE-NEXT:               %[[MASKCOND:.*]] = icmp eq i64 %[[MASKEDPTR]], 0
  // CHECK-SANITIZE-NEXT:               %[[PTRINT_DUP:.*]] = ptrtoint i8** %[[X_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[MASKCOND]], label %[[CONT:.*]], label %[[HANDLER_ALIGNMENT_ASSUMPTION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_ALIGNMENT_ASSUMPTION]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_alignment_assumption_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}, {{{.*}}}* }* @[[LINE_100_ALIGNMENT_ASSUMPTION]] to i8*), i64 %[[PTRINT_DUP]], i64 2147483648, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_alignment_assumption(i8* bitcast ({ {{{.*}}}, {{{.*}}}, {{{.*}}}* }* @[[LINE_100_ALIGNMENT_ASSUMPTION]] to i8*), i64 %[[PTRINT_DUP]], i64 2147483648, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-SANITIZE-NEXT:               call void @llvm.assume(i1 %[[MASKCOND]])
  // CHECK-NEXT:                        ret i8** %[[X_RELOADED]]
  // CHECK-NEXT:                      }
#line 100
  return x;
}
