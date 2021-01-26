// RUN: %clang_cc1 -fopenmp-simd -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -fsanitize=alignment -fno-sanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -fopenmp-simd -fsanitize=alignment -fsanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fopenmp-simd -fsanitize=alignment -fsanitize-trap=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER: @[[CHAR:.*]] = {{.*}} c"'char *'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_100_ALIGNMENT_ASSUMPTION:.*]] = {{.*}}, i32 100, i32 30 }, {{.*}}* @[[CHAR]] }

void func(char *data) {
  // CHECK: define{{.*}} void @{{.*}}(i8* %[[DATA:.*]])
  // CHECK-NEXT: [[ENTRY:.*]]:
  // CHECK-NEXT:   %[[DATA_ADDR:.*]] = alloca i8*, align 8
  // CHECK:   store i8* %[[DATA]], i8** %[[DATA_ADDR]], align 8
  // CHECK:   %[[DATA_RELOADED:.*]] = load i8*, i8** %[[DATA_ADDR]], align 8
  // CHECK-SANITIZE-NEXT:   %[[PTRINT:.*]] = ptrtoint i8* %[[DATA_RELOADED]] to i64
  // CHECK-SANITIZE-NEXT:   %[[MASKEDPTR:.*]] = and i64 %[[PTRINT]], 1073741823
  // CHECK-SANITIZE-NEXT:   %[[MASKCOND:.*]] = icmp eq i64 %[[MASKEDPTR]], 0
  // CHECK-SANITIZE-NEXT:               %[[PTRINT_DUP:.*]] = ptrtoint i8* %[[DATA_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[MASKCOND]], label %[[CONT:.*]], label %[[HANDLER_ALIGNMENT_ASSUMPTION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_ALIGNMENT_ASSUMPTION]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_alignment_assumption_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}, {{{.*}}}* }* @[[LINE_100_ALIGNMENT_ASSUMPTION]] to i8*), i64 %[[PTRINT_DUP]], i64 1073741824, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_alignment_assumption(i8* bitcast ({ {{{.*}}}, {{{.*}}}, {{{.*}}}* }* @[[LINE_100_ALIGNMENT_ASSUMPTION]] to i8*), i64 %[[PTRINT_DUP]], i64 1073741824, i64 0){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 23){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        call void @llvm.assume(i1 true) [ "align"(i8* %[[DATA_RELOADED]], i64 1073741824) ]

#line 100
#pragma omp for simd aligned(data : 0x40000000)
  for (int x = 0; x < 1; x++)
    data[x] = data[x];
}
