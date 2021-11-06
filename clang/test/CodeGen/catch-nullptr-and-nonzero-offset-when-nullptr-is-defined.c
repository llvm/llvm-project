// RUN: %clang_cc1 -x c -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLNOTOK,CHECK-SANITIZE-NULLNOTOK-C,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLNOTOK,CHECK-SANITIZE-NULLNOTOK-C,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLNOTOK,CHECK-SANITIZE-NULLNOTOK-C,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// RUN: %clang_cc1 -x c -fno-delete-null-pointer-checks -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -x c -fno-delete-null-pointer-checks -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLOK,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -x c -fno-delete-null-pointer-checks -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLOK,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -x c -fno-delete-null-pointer-checks -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLOK,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// RUN: %clang_cc1 -x c++ -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLNOTOK,CHECK-SANITIZE-NULLNOTOK-CPP,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLNOTOK,CHECK-SANITIZE-NULLNOTOK-CPP,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLNOTOK,CHECK-SANITIZE-NULLNOTOK-CPP,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// RUN: %clang_cc1 -x c++ -fno-delete-null-pointer-checks -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -x c++ -fno-delete-null-pointer-checks -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLOK,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -x c++ -fno-delete-null-pointer-checks -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLOK,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -x c++ -fno-delete-null-pointer-checks -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-NULLOK,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_100:.*]] = {{.*}}, i32 100, i32 15 } }

#ifdef __cplusplus
extern "C" {
#endif

char *add_unsigned(char *base, unsigned long offset) {
  // CHECK:                           define{{.*}} i8* @add_unsigned(i8* noundef %[[BASE:.*]], i64 noundef %[[OFFSET:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca i8*, align 8
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store i8* %[[BASE]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BASE_RELOADED]], i64 %[[OFFSET_RELOADED]]
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_AGGREGATE:.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %[[OFFSET_RELOADED]]), !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_OVERFLOWED:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint i8* %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], %[[COMPUTED_OFFSET]], !nosanitize
  // CHECK-SANITIZE-NULLNOTOK-NEXT:     %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne i8* %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NULLNOTOK-NEXT:     %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-NULLNOTOK-C-NEXT:   %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NULLNOTOK-CPP-NEXT: %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[COMPUTED_OFFSET_OVERFLOWED]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NULLNOTOK-NEXT:     %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NULLNOTOK-NEXT:     br i1 %[[GEP_IS_OKAY]],
  // CHECK-SANITIZE-NULLOK-NEXT:        br i1 %[[GEP_DID_NOT_OVERFLOW]],
  // CHECK-SANITIZE-SAME:                                                label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_100]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_100]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR]]
#line 100
  return base + offset;
}

#ifdef __cplusplus
}
#endif
