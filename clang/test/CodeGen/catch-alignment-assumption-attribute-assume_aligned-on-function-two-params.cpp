// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-NOSANITIZE
// RUN: %clang_cc1 -fsanitize=alignment -fno-sanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-trap=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER: @[[CHAR:.*]] = {{.*}} c"'char **'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_100_ALIGNMENT_ASSUMPTION:.*]] = {{.*}}, i32 100, i32 10 }, {{.*}}* @[[CHAR]] }

char **__attribute__((assume_aligned(0x80000000, 42))) passthrough(char **x) {
  // CHECK:      define{{.*}} i8** @[[PASSTHROUGH:.*]](i8** %[[X:.*]])
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %[[X_ADDR:.*]] = alloca i8**, align 8
  // CHECK-NEXT:   store i8** %[[X]], i8*** %[[X_ADDR]], align 8
  // CHECK-NEXT:   %[[X_RELOADED:.*]] = load i8**, i8*** %[[X_ADDR]], align 8
  // CHECK-NEXT:   ret i8** %[[X_RELOADED]]
  // CHECK-NEXT: }
  return x;
}

char **caller(char **x) {
  // CHECK:                           define{{.*}} i8** @{{.*}}(i8** %[[X:.*]])
  // CHECK-NEXT:                      entry:
  // CHECK-NEXT:                        %[[X_ADDR:.*]] = alloca i8**, align 8
  // CHECK-NEXT:                        store i8** %[[X]], i8*** %[[X_ADDR]], align 8
  // CHECK-NEXT:                        %[[X_RELOADED:.*]] = load i8**, i8*** %[[X_ADDR]], align 8
  // CHECK-NEXT:                        %[[X_RETURNED:.*]] = call i8** @[[PASSTHROUGH]](i8** %[[X_RELOADED]])
  // CHECK-SANITIZE-NEXT:               %[[PTRINT:.*]] = ptrtoint i8** %[[X_RETURNED]] to i64
  // CHECK-SANITIZE-NEXT:               %[[OFFSETPTR:.*]] = sub i64 %[[PTRINT]], 42
  // CHECK-SANITIZE-NEXT:               %[[MASKEDPTR:.*]] = and i64 %[[OFFSETPTR]], 2147483647
  // CHECK-SANITIZE-NEXT:               %[[MASKCOND:.*]] = icmp eq i64 %[[MASKEDPTR]], 0
  // CHECK-SANITIZE-NEXT:               %[[PTRINT_DUP:.*]] = ptrtoint i8** %[[X_RETURNED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[MASKCOND]], label %[[CONT:.*]], label %[[HANDLER_ALIGNMENT_ASSUMPTION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_ALIGNMENT_ASSUMPTION]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_alignment_assumption_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}, {{{.*}}}* }* @[[LINE_100_ALIGNMENT_ASSUMPTION]] to i8*), i64 %[[PTRINT_DUP]], i64 2147483648, i64 42){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_alignment_assumption(i8* bitcast ({ {{{.*}}}, {{{.*}}}, {{{.*}}}* }* @[[LINE_100_ALIGNMENT_ASSUMPTION]] to i8*), i64 %[[PTRINT_DUP]], i64 2147483648, i64 42){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 23){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        call void @llvm.assume(i1 true) [ "align"(i8** %[[X_RETURNED]], i64 2147483648, i64 42) ]
  // CHECK-NEXT:                        ret i8** %[[X_RETURNED]]
  // CHECK-NEXT:                      }
#line 100
  return passthrough(x);
}
