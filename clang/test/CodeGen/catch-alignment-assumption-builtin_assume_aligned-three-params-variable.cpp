// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-NOSANITIZE
// RUN: %clang_cc1 -fsanitize=alignment -fno-sanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-trap=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER: @[[CHAR:.*]] = {{.*}} c"'char **'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_100_ALIGNMENT_ASSUMPTION:.*]] = {{.*}}, i32 100, i32 35 }, {{.*}}* @[[CHAR]] }

void *caller(char **x, unsigned long offset) {
  // CHECK:                           define i8* @{{.*}}(i8** %[[X:.*]], i64 %[[OFFSET:.*]])
  // CHECK-NEXT:                      entry:
  // CHECK-NEXT:                        %[[X_ADDR:.*]] = alloca i8**, align 8
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store i8** %[[X]], i8*** %[[X_ADDR]], align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[X_RELOADED:.*]] = load i8**, i8*** %[[X_ADDR]], align 8
  // CHECK-NEXT:                        %[[BITCAST:.*]] = bitcast i8** %[[X_RELOADED]] to i8*
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[PTRINT:.*]] = ptrtoint i8* %[[BITCAST]] to i64
  // CHECK-NEXT:                        %[[OFFSETPTR:.*]] = sub i64 %[[PTRINT]], %[[OFFSET_RELOADED]]
  // CHECK-NEXT:                        %[[MASKEDPTR:.*]] = and i64 %[[OFFSETPTR]], 2147483647
  // CHECK-NEXT:                        %[[MASKCOND:.*]] = icmp eq i64 %[[MASKEDPTR]], 0
  // CHECK-SANITIZE-NEXT:               %[[PTRINT_DUP:.*]] = ptrtoint i8* %[[BITCAST]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[MASKCOND]], label %[[CONT:.*]], label %[[HANDLER_ALIGNMENT_ASSUMPTION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_ALIGNMENT_ASSUMPTION]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_alignment_assumption_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}, {{{.*}}}* }* @[[LINE_100_ALIGNMENT_ASSUMPTION]] to i8*), i64 %[[PTRINT_DUP]], i64 2147483648, i64 %[[OFFSET_RELOADED]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_alignment_assumption(i8* bitcast ({ {{{.*}}}, {{{.*}}}, {{{.*}}}* }* @[[LINE_100_ALIGNMENT_ASSUMPTION]] to i8*), i64 %[[PTRINT_DUP]], i64 2147483648, i64 %[[OFFSET_RELOADED]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        call void @llvm.assume(i1 %[[MASKCOND]])
  // CHECK-NEXT:                        ret i8* %[[BITCAST]]
  // CHECK-NEXT:                      }
#line 100
  return __builtin_assume_aligned(x, 0x80000000, offset);
}
