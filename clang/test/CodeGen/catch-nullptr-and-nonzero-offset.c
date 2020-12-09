// RUN: %clang_cc1 -x c -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-NOSANITIZE
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE,CHECK-SANITIZE-C,CHECK-SANITIZE-ANYRECOVER-C,CHECK-SANITIZE-NORECOVER-C,CHECK-SANITIZE-UNREACHABLE-C
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER,CHECK-SANITIZE-C,CHECK-SANITIZE-ANYRECOVER-C,CHECK-SANITIZE-RECOVER-C
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE,CHECK-SANITIZE-C,CHECK-SANITIZE-TRAP-C,CHECK-SANITIZE-UNREACHABLE-C

// RUN: %clang_cc1 -x c++ -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-NOSANITIZE
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE,CHECK-SANITIZE-CPP,CHECK-SANITIZE-ANYRECOVER-CPP,CHECK-SANITIZE-NORECOVER-CPP,CHECK-SANITIZE-UNREACHABLE-CPP
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER,CHECK-SANITIZE-CPP,CHECK-SANITIZE-ANYRECOVER-CPP,CHECK-SANITIZE-RECOVER-CPP
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE,CHECK-SANITIZE-CPP,CHECK-SANITIZE-TRAP-CPP,CHECK-SANITIZE-UNREACHABLE-CPP

// In C++/LLVM IR, if the base pointer evaluates to a null pointer value,
// the only valid pointer this inbounds GEP can produce is also a null pointer.
// Likewise, if we have non-zero base pointer, we can not get null pointer
// as a result, so the offset can not be -int(BasePtr).

// So in other words, the offset can not change "null status" of the pointer,
// as in if the pointer was null, it can not become non-null, and vice versa,
// if it was non-null, it can not become null.

// In C, however, offsetting null pointer is completely undefined, even by 0.

// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_100:.*]] = {{.*}}, i32 100, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-C-DAG: @[[LINE_200:.*]] = {{.*}}, i32 200, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_300:.*]] = {{.*}}, i32 300, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_400:.*]] = {{.*}}, i32 400, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_500:.*]] = {{.*}}, i32 500, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-C-DAG: @[[LINE_600:.*]] = {{.*}}, i32 600, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_700:.*]] = {{.*}}, i32 700, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_800:.*]] = {{.*}}, i32 800, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_900:.*]] = {{.*}}, i32 900, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-C-DAG: @[[LINE_1000:.*]] = {{.*}}, i32 1000, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_1100:.*]] = {{.*}}, i32 1100, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_1200:.*]] = {{.*}}, i32 1200, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_1300:.*]] = {{.*}}, i32 1300, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-C-DAG: @[[LINE_1400:.*]] = {{.*}}, i32 1400, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_1500:.*]] = {{.*}}, i32 1500, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_1600:.*]] = {{.*}}, i32 1600, i32 15 } }

#ifdef __cplusplus
extern "C" {
#endif

char *var_var(char *base, unsigned long offset) {
  // CHECK: define i8* @var_var(i8* %[[BASE:.*]], i64 %[[OFFSET:.*]])
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
  // CHECK-SANITIZE-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne i8* %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:             %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[COMPUTED_OFFSET_OVERFLOWED]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
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

char *var_zero(char *base) {
  // CHECK:                             define i8* @var_zero(i8* %[[BASE:.*]])
  // CHECK-NEXT:                        [[ENTRY:.*]]:
  // CHECK-NEXT:                          %[[BASE_ADDR:.*]] = alloca i8*, align 8
  // CHECK-NEXT:                          store i8* %[[BASE]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                          %[[BASE_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                          %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BASE_RELOADED]], i64 0
  // CHECK-SANITIZE-C-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint i8* %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-C-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne i8* %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-C-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:             %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-C-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-C-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[COMPUTED_GEP_IS_UGE_BASE]], !nosanitize
  // CHECK-SANITIZE-C-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE-C:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-C-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_200]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-C-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_200]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-C-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-C-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE-C:                  [[CONT]]:
  // CHECK-NEXT:                          ret i8* %[[ADD_PTR]]
  static const unsigned long offset = 0;
#line 200
  return base + offset;
}

char *var_one(char *base) {
  // CHECK:                           define i8* @var_one(i8* %[[BASE:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca i8*, align 8
  // CHECK-NEXT:                        store i8* %[[BASE]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BASE_RELOADED]], i64 1
  // CHECK-SANITIZE-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint i8* %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne i8* %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:             %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[COMPUTED_GEP_IS_UGE_BASE]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_300]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_300]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR]]
  static const unsigned long offset = 1;
#line 300
  return base + offset;
}

char *var_allones(char *base) {
  // CHECK:                           define i8* @var_allones(i8* %[[BASE:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca i8*, align 8
  // CHECK-NEXT:                        store i8* %[[BASE]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BASE_RELOADED]], i64 -1
  // CHECK-SANITIZE-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint i8* %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], -1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne i8* %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:             %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[COMPUTED_GEP_IS_UGE_BASE]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_400]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_400]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR]]
  static const unsigned long offset = -1;
#line 400
  return base + offset;
}

//------------------------------------------------------------------------------

char *nullptr_var(unsigned long offset) {
  // CHECK:                           define i8* @nullptr_var(i64 %[[OFFSET:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* null, i64 %[[OFFSET_RELOADED]]
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_AGGREGATE:.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %[[OFFSET_RELOADED]]), !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_OVERFLOWED:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 0, %[[COMPUTED_OFFSET]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:             %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 false, %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 false, %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[COMPUTED_OFFSET_OVERFLOWED]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_500]] to i8*), i64 0, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_500]] to i8*), i64 0, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR]]
  static char *const base = (char *)0;
#line 500
  return base + offset;
}

char *nullptr_zero() {
  // CHECK:                             define i8* @nullptr_zero()
  // CHECK-NEXT:                        [[ENTRY:.*]]:
  // CHECK-SANITIZE-C-NEXT:               br i1 false, label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE-C:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-C-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_600]] to i8*), i64 0, i64 0)
  // CHECK-SANITIZE-RECOVER-C-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_600]] to i8*), i64 0, i64 0)
  // CHECK-SANITIZE-TRAP-C-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-C-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE-C:                  [[CONT]]:
  // CHECK-NEXT:                          ret i8* null
  static char *const base = (char *)0;
  static const unsigned long offset = 0;
#line 600
  return base + offset;
}

char *nullptr_one_BAD() {
  // CHECK:                           define i8* @nullptr_one_BAD()
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-SANITIZE-C-NEXT:             br i1 false, label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           br i1 icmp eq (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* null, i64 1) to i64), i64 0), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_700]] to i8*), i64 0, i64 ptrtoint (i8* getelementptr inbounds (i8, i8* null, i64 1) to i64))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_700]] to i8*), i64 0, i64 ptrtoint (i8* getelementptr inbounds (i8, i8* null, i64 1) to i64))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* getelementptr inbounds (i8, i8* null, i64 1)
  static char *const base = (char *)0;
  static const unsigned long offset = 1;
#line 700
  return base + offset;
}

char *nullptr_allones_BAD() {
  // CHECK:                           define i8* @nullptr_allones_BAD()
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-SANITIZE-C-NEXT:             br i1 false, label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           br i1 icmp eq (i64 mul (i64 ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64), i64 -1), i64 0), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_800]] to i8*), i64 0, i64 mul (i64 ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64), i64 -1))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_800]] to i8*), i64 0, i64 mul (i64 ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64), i64 -1))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* getelementptr inbounds (i8, i8* null, i64 -1)
  static char *const base = (char *)0;
  static const unsigned long offset = -1;
#line 800
  return base + offset;
}

//------------------------------------------------------------------------------

char *one_var(unsigned long offset) {
  // CHECK:                           define i8* @one_var(i64 %[[OFFSET:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* inttoptr (i64 1 to i8*), i64 %[[OFFSET_RELOADED]]
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_AGGREGATE:.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %[[OFFSET_RELOADED]]), !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_OVERFLOWED:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 1, %[[COMPUTED_OFFSET]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:             %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 icmp ne (i8* inttoptr (i64 1 to i8*), i8* null), %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 icmp ne (i8* inttoptr (i64 1 to i8*), i8* null), %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[COMPUTED_OFFSET_OVERFLOWED]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_900]] to i8*), i64 1, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_900]] to i8*), i64 1, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR]]
  static char *const base = (char *)1;
#line 900
  return base + offset;
}

char *one_zero() {
  // CHECK:                             define i8* @one_zero()
  // CHECK-NEXT:                        [[ENTRY:.*]]:
  // CHECK-SANITIZE-C-NEXT:               br i1 icmp ne (i8* inttoptr (i64 1 to i8*), i8* null), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE-C:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-C-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_1000]] to i8*), i64 1, i64 1)
  // CHECK-SANITIZE-RECOVER-C-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_1000]] to i8*), i64 1, i64 1)
  // CHECK-SANITIZE-TRAP-C-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-C-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE-C:                  [[CONT]]:
  // CHECK-NEXT:                          ret i8* inttoptr (i64 1 to i8*)
  static char *const base = (char *)1;
  static const unsigned long offset = 0;
#line 1000
  return base + offset;
}

char *one_one_OK() {
  // CHECK:                           define i8* @one_one_OK()
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-SANITIZE-C-NEXT:             br i1 and (i1 icmp ne (i8* inttoptr (i64 1 to i8*), i8* null), i1 icmp ne (i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 1 to i8*), i64 1) to i64), i64 1), i64 1), i64 0)), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           br i1 xor (i1 icmp eq (i8* inttoptr (i64 1 to i8*), i8* null), i1 icmp ne (i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 1 to i8*), i64 1) to i64), i64 1), i64 1), i64 0)), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_1100]] to i8*), i64 1, i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 1 to i8*), i64 1) to i64), i64 1), i64 1))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_1100]] to i8*), i64 1, i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 1 to i8*), i64 1) to i64), i64 1), i64 1))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* getelementptr inbounds (i8, i8* inttoptr (i64 1 to i8*), i64 1)
  static char *const base = (char *)1;
  static const unsigned long offset = 1;
#line 1100
  return base + offset;
}

char *one_allones_BAD() {
  // CHECK:                           define i8* @one_allones_BAD()
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-SANITIZE-C-NEXT:             br i1 and (i1 icmp ne (i8* inttoptr (i64 1 to i8*), i8* null), i1 icmp ne (i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 1 to i8*), i64 -1) to i64), i64 1), i64 1), i64 0)), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           br i1 xor (i1 icmp eq (i8* inttoptr (i64 1 to i8*), i8* null), i1 icmp ne (i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 1 to i8*), i64 -1) to i64), i64 1), i64 1), i64 0)), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_1200]] to i8*), i64 1, i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 1 to i8*), i64 -1) to i64), i64 1), i64 1))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_1200]] to i8*), i64 1, i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 1 to i8*), i64 -1) to i64), i64 1), i64 1))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* getelementptr inbounds (i8, i8* inttoptr (i64 1 to i8*), i64 -1)
  static char *const base = (char *)1;
  static const unsigned long offset = -1;
#line 1200
  return base + offset;
}

//------------------------------------------------------------------------------

char *allones_var(unsigned long offset) {
  // CHECK:                           define i8* @allones_var(i64 %[[OFFSET:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* inttoptr (i64 -1 to i8*), i64 %[[OFFSET_RELOADED]]
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_AGGREGATE:.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %[[OFFSET_RELOADED]]), !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_OVERFLOWED:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 -1, %[[COMPUTED_OFFSET]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:             %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 icmp ne (i8* inttoptr (i64 -1 to i8*), i8* null), %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 icmp ne (i8* inttoptr (i64 -1 to i8*), i8* null), %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[COMPUTED_OFFSET_OVERFLOWED]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], -1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_1300]] to i8*), i64 -1, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_1300]] to i8*), i64 -1, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR]]
  static char *const base = (char *)-1;
#line 1300
  return base + offset;
}

char *allones_zero_OK() {
  // CHECK:                             define i8* @allones_zero_OK()
  // CHECK-NEXT:                        [[ENTRY:.*]]:
  // CHECK-SANITIZE-C-NEXT:               br i1 icmp ne (i8* inttoptr (i64 -1 to i8*), i8* null), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE-C:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-C-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_1400]] to i8*), i64 -1, i64 -1)
  // CHECK-SANITIZE-RECOVER-C-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_1400]] to i8*), i64 -1, i64 -1)
  // CHECK-SANITIZE-TRAP-C-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-C-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE-C:                  [[CONT]]:
  // CHECK-NEXT:                          ret i8* inttoptr (i64 -1 to i8*)
  static char *const base = (char *)-1;
  static const unsigned long offset = 0;
#line 1400
  return base + offset;
}

char *allones_one_BAD() {
  // CHECK: define i8* @allones_one_BAD()
  // CHECK-NEXT: [[ENTRY:.*]]:
  // CHECK-SANITIZE-C-NEXT:             br i1 and (i1 icmp ne (i8* inttoptr (i64 -1 to i8*), i8* null), i1 icmp ne (i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 -1 to i8*), i64 1) to i64), i64 -1), i64 -1), i64 0)), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           br i1 xor (i1 icmp eq (i8* inttoptr (i64 -1 to i8*), i8* null), i1 icmp ne (i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 -1 to i8*), i64 1) to i64), i64 -1), i64 -1), i64 0)), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_1500]] to i8*), i64 -1, i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 -1 to i8*), i64 1) to i64), i64 -1), i64 -1))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_1500]] to i8*), i64 -1, i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 -1 to i8*), i64 1) to i64), i64 -1), i64 -1))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* getelementptr inbounds (i8, i8* inttoptr (i64 -1 to i8*), i64 1)
  static char *const base = (char *)-1;
  static const unsigned long offset = 1;
#line 1500
  return base + offset;
}

char *allones_allones_OK() {
  // CHECK: define i8* @allones_allones_OK()
  // CHECK-NEXT: [[ENTRY:.*]]:
  // CHECK-SANITIZE-C-NEXT:             br i1 and (i1 icmp ne (i8* inttoptr (i64 -1 to i8*), i8* null), i1 icmp ne (i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 -1 to i8*), i64 -1) to i64), i64 -1), i64 -1), i64 0)), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           br i1 xor (i1 icmp eq (i8* inttoptr (i64 -1 to i8*), i8* null), i1 icmp ne (i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 -1 to i8*), i64 -1) to i64), i64 -1), i64 -1), i64 0)), label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_1600]] to i8*), i64 -1, i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 -1 to i8*), i64 -1) to i64), i64 -1), i64 -1))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_1600]] to i8*), i64 -1, i64 add (i64 sub (i64 ptrtoint (i8* getelementptr inbounds (i8, i8* inttoptr (i64 -1 to i8*), i64 -1) to i64), i64 -1), i64 -1))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* getelementptr inbounds (i8, i8* inttoptr (i64 -1 to i8*), i64 -1)
  static char *const base = (char *)-1;
  static const unsigned long offset = -1;
#line 1600
  return base + offset;
}

#ifdef __cplusplus
}
#endif
