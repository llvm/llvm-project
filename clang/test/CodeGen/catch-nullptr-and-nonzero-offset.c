// RUN: %clang_cc1 -x c -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// RUN: %clang_cc1 -x c++ -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// In C++/LLVM IR, if the base pointer evaluates to a null pointer value,
// the only valid pointer this inbounds GEP can produce is also a null pointer.
// Likewise, if we have non-zero base pointer, we can not get null pointer
// as a result, so the offset can not be -int(BasePtr).

// So in other words, the offset can not change "null status" of the pointer,
// as in if the pointer was null, it can not become non-null, and vice versa,
// if it was non-null, it can not become null.

// In C, however, offsetting null pointer is completely undefined, even by 0.

// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_100:.*]] = {{.*}}, i32 100, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_300:.*]] = {{.*}}, i32 300, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_400:.*]] = {{.*}}, i32 400, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_500:.*]] = {{.*}}, i32 500, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_700:.*]] = {{.*}}, i32 700, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_800:.*]] = {{.*}}, i32 800, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_900:.*]] = {{.*}}, i32 900, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_1100:.*]] = {{.*}}, i32 1100, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_1200:.*]] = {{.*}}, i32 1200, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_1300:.*]] = {{.*}}, i32 1300, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_1500:.*]] = {{.*}}, i32 1500, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_1600:.*]] = {{.*}}, i32 1600, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_1700:.*]] = {{.*}}, i32 1700, i32 15 } }

#ifdef __cplusplus
extern "C" {
#endif

char *var_var(char *base, unsigned long offset) {
  // CHECK: define{{.*}} ptr @var_var(ptr noundef %[[BASE:.*]], i64 noundef %[[OFFSET:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca ptr, align 8
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store ptr %[[BASE]], ptr %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], ptr %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load ptr, ptr %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, ptr %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds nuw i8, ptr %[[BASE_RELOADED]], i64 %[[OFFSET_RELOADED]]
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_AGGREGATE:.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %[[OFFSET_RELOADED]]), !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_OVERFLOWED:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[OR_OV:.+]] = or i1 %[[COMPUTED_OFFSET_OVERFLOWED]], false, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint ptr %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], %[[COMPUTED_OFFSET]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne ptr %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[OR_OV]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_100]], i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_100]], i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr %[[ADD_PTR]]
#line 100
  return base + offset;
}

char *var_zero(char *base) {
  // CHECK:                             define{{.*}} ptr @var_zero(ptr noundef %[[BASE:.*]])
  // CHECK-NEXT:                        [[ENTRY:.*]]:
  // CHECK-NEXT:                          %[[BASE_ADDR:.*]] = alloca ptr, align 8
  // CHECK-NEXT:                          store ptr %[[BASE]], ptr %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                          %[[BASE_RELOADED:.*]] = load ptr, ptr %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                          %[[ADD_PTR:.*]] = getelementptr inbounds nuw i8, ptr %[[BASE_RELOADED]], i64 0
  // CHECK-NEXT:                          ret ptr %[[ADD_PTR]]
  static const unsigned long offset = 0;
#line 200
  return base + offset;
}

char *var_one(char *base) {
  // CHECK:                           define{{.*}} ptr @var_one(ptr noundef %[[BASE:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca ptr, align 8
  // CHECK-NEXT:                        store ptr %[[BASE]], ptr %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load ptr, ptr %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds nuw i8, ptr %[[BASE_RELOADED]], i64 1
  // CHECK-SANITIZE-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint ptr %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne ptr %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[AND_TRUE:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[AND_TRUE]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_300]], i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_300]], i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr %[[ADD_PTR]]
  static const unsigned long offset = 1;
#line 300
  return base + offset;
}

char *var_allones(char *base) {
  // CHECK:                           define{{.*}} ptr @var_allones(ptr noundef %[[BASE:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca ptr, align 8
  // CHECK-NEXT:                        store ptr %[[BASE]], ptr %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load ptr, ptr %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds nuw i8, ptr %[[BASE_RELOADED]], i64 -1
  // CHECK-SANITIZE-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint ptr %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], -1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne ptr %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[AND_TRUE:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[AND_TRUE]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_400]], i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_400]], i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr %[[ADD_PTR]]
  static const unsigned long offset = -1;
#line 400
  return base + offset;
}

//------------------------------------------------------------------------------

char *nullptr_var(unsigned long offset) {
  // CHECK:                           define{{.*}} ptr @nullptr_var(i64 noundef %[[OFFSET:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], ptr %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, ptr %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds nuw i8, ptr null, i64 %[[OFFSET_RELOADED]]
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_AGGREGATE:.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %[[OFFSET_RELOADED]]), !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_OVERFLOWED:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[OR_OV:.+]] = or i1 %[[COMPUTED_OFFSET_OVERFLOWED]], false, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 0, %[[COMPUTED_OFFSET]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 false, %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[OR_OV]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_500]], i64 0, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_500]], i64 0, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr %[[ADD_PTR]]
  static char *const base = (char *)0;
#line 500
  return base + offset;
}

char *nullptr_zero(void) {
  // CHECK:                             define{{.*}} ptr @nullptr_zero()
  // CHECK-NEXT:                        [[ENTRY:.*]]:
  // CHECK-NEXT:                          ret ptr null
  static char *const base = (char *)0;
  static const unsigned long offset = 0;
#line 600
  return base + offset;
}

char *nullptr_one_BAD(void) {
  // CHECK:                           define{{.*}} ptr @nullptr_one_BAD()
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-SANITIZE-NEXT:             %[[CMP:.*]] = icmp ne i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr null, i64 1) to i64), 0, !nosanitize
  // CHECK-SANITIZE-NEXT:             %[[COND:.*]] = icmp eq i1 false, %[[CMP]], !nosanitize
  // CHECK-SANITIZE-NEXT:             br i1 %[[COND]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_700]], i64 0, i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr null, i64 1) to i64))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_700]], i64 0, i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr null, i64 1) to i64))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr getelementptr inbounds nuw (i8, ptr null, i64 1)
  static char *const base = (char *)0;
  static const unsigned long offset = 1;
#line 700
  return base + offset;
}

char *nullptr_allones_BAD(void) {
  // CHECK:                           define{{.*}} ptr @nullptr_allones_BAD()
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-SANITIZE-NEXT:             %[[CMP:.*]] = icmp ne i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr null, i64 -1) to i64), 0, !nosanitize
  // CHECK-SANITIZE-NEXT:             %[[COND:.*]] = icmp eq i1 false, %[[CMP]], !nosanitize
  // CHECK-SANITIZE-NEXT:             br i1 %[[COND]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_800]], i64 0, i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr null, i64 -1) to i64))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_800]], i64 0, i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr null, i64 -1) to i64))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr getelementptr inbounds nuw (i8, ptr null, i64 -1)
  static char *const base = (char *)0;
  static const unsigned long offset = -1;
#line 800
  return base + offset;
}

//------------------------------------------------------------------------------

char *one_var(unsigned long offset) {
  // CHECK:                           define{{.*}} ptr @one_var(i64 noundef %[[OFFSET:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], ptr %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, ptr %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds nuw i8, ptr inttoptr (i64 1 to ptr), i64 %[[OFFSET_RELOADED]]
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_AGGREGATE:.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %[[OFFSET_RELOADED]]), !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_OVERFLOWED:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[OR_OV:.+]] = or i1 %[[COMPUTED_OFFSET_OVERFLOWED]], false, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 1, %[[COMPUTED_OFFSET]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[OTHER_IS_NOT_NULL:.*]] = icmp ne ptr inttoptr (i64 1 to ptr), null
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[OTHER_IS_NOT_NULL]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[OR_OV]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_900]], i64 1, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_900]], i64 1, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr %[[ADD_PTR]]
  static char *const base = (char *)1;
#line 900
  return base + offset;
}

char *one_zero(void) {
  // CHECK:                             define{{.*}} ptr @one_zero()
  // CHECK-NEXT:                        [[ENTRY:.*]]:
  // CHECK-NEXT:                          ret ptr inttoptr (i64 1 to ptr)
  static char *const base = (char *)1;
  static const unsigned long offset = 0;
#line 1000
  return base + offset;
}

char *one_one_OK(void) {
  // CHECK:                           define{{.*}} ptr @one_one_OK()
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-SANITIZE-NEXT:               %[[CMP1:.*]] = icmp ne ptr inttoptr (i64 1 to ptr), null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[CMP2:.*]] = icmp ne i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 1 to ptr), i64 1) to i64), i64 1), i64 1), 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COND:.*]] = icmp eq i1 %[[CMP1]], %[[CMP2]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[COND]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_1100]], i64 1, i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 1 to ptr), i64 1) to i64), i64 1), i64 1))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_1100]], i64 1, i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 1 to ptr), i64 1) to i64), i64 1), i64 1))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 1 to ptr), i64 1)
  static char *const base = (char *)1;
  static const unsigned long offset = 1;
#line 1100
  return base + offset;
}

char *one_allones_BAD(void) {
  // CHECK:                           define{{.*}} ptr @one_allones_BAD()
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-SANITIZE-NEXT:               %[[CMP1:.*]] = icmp ne ptr inttoptr (i64 1 to ptr), null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[CMP2:.*]] = icmp ne i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 1 to ptr), i64 -1) to i64), i64 1), i64 1), 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COND:.*]] = icmp eq i1 %[[CMP1]], %[[CMP2]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[COND]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_1200]], i64 1, i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 1 to ptr), i64 -1) to i64), i64 1), i64 1))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_1200]], i64 1, i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 1 to ptr), i64 -1) to i64), i64 1), i64 1))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 1 to ptr), i64 -1)
  static char *const base = (char *)1;
  static const unsigned long offset = -1;
#line 1200
  return base + offset;
}

//------------------------------------------------------------------------------

char *allones_var(unsigned long offset) {
  // CHECK:                           define{{.*}} ptr @allones_var(i64 noundef %[[OFFSET:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], ptr %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, ptr %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds nuw i8, ptr inttoptr (i64 -1 to ptr), i64 %[[OFFSET_RELOADED]]
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_AGGREGATE:.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %[[OFFSET_RELOADED]]), !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_OVERFLOWED:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[OR_OV:.+]] = or i1 %[[COMPUTED_OFFSET_OVERFLOWED]], false, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 -1, %[[COMPUTED_OFFSET]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[OTHER_IS_NOT_NULL:.*]] = icmp ne ptr inttoptr (i64 -1 to ptr), null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[OTHER_IS_NOT_NULL]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[OR_OV]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], -1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_1300]], i64 -1, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_1300]], i64 -1, i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr %[[ADD_PTR]]
  static char *const base = (char *)-1;
#line 1300
  return base + offset;
}

char *allones_zero_OK(void) {
  // CHECK:                             define{{.*}} ptr @allones_zero_OK()
  // CHECK-NEXT:                        [[ENTRY:.*]]:
  // CHECK-NEXT:                          ret ptr inttoptr (i64 -1 to ptr)
  static char *const base = (char *)-1;
  static const unsigned long offset = 0;
#line 1400
  return base + offset;
}

char *allones_one_BAD(void) {
  // CHECK: define{{.*}} ptr @allones_one_BAD()
  // CHECK-NEXT: [[ENTRY:.*]]:
  // CHECK-SANITIZE-NEXT:               %[[CMP1:.*]] = icmp ne ptr inttoptr (i64 -1 to ptr), null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[CMP2:.*]] = icmp ne i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 -1 to ptr), i64 1) to i64), i64 -1), i64 -1), 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COND:.*]] = icmp eq i1 %[[CMP1]], %[[CMP2]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[COND]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_1500]], i64 -1, i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 -1 to ptr), i64 1) to i64), i64 -1), i64 -1))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_1500]], i64 -1, i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 -1 to ptr), i64 1) to i64), i64 -1), i64 -1))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 -1 to ptr), i64 1)
  static char *const base = (char *)-1;
  static const unsigned long offset = 1;
#line 1500
  return base + offset;
}

char *allones_allones_OK(void) {
  // CHECK: define{{.*}} ptr @allones_allones_OK()
  // CHECK-NEXT: [[ENTRY:.*]]:
  // CHECK-SANITIZE-NEXT:               %[[CMP1:.*]] = icmp ne ptr inttoptr (i64 -1 to ptr), null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[CMP2:.*]] = icmp ne i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 -1 to ptr), i64 -1) to i64), i64 -1), i64 -1), 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COND:.*]] = icmp eq i1 %[[CMP1]], %[[CMP2]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[COND]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_1600]], i64 -1, i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 -1 to ptr), i64 -1) to i64), i64 -1), i64 -1))
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_1600]], i64 -1, i64 add (i64 sub (i64 ptrtoint (ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 -1 to ptr), i64 -1) to i64), i64 -1), i64 -1))
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr getelementptr inbounds nuw (i8, ptr inttoptr (i64 -1 to ptr), i64 -1)
  static char *const base = (char *)-1;
  static const unsigned long offset = -1;
#line 1600
  return base + offset;
}

// C++ does not allow void* arithmetic even as a GNU extension. Replace void*
// with char* in that case to keep test expectations the same.
#ifdef __cplusplus
char *void_ptr(char *base, unsigned long offset) {
#else
char *void_ptr(void *base, unsigned long offset) {
#endif
  // CHECK: define{{.*}} ptr @void_ptr(ptr noundef %[[BASE:.*]], i64 noundef %[[OFFSET:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca ptr, align 8
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store ptr %[[BASE]], ptr %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], ptr %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load ptr, ptr %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, ptr %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds nuw i8, ptr %[[BASE_RELOADED]], i64 %[[OFFSET_RELOADED]]
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_AGGREGATE:.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %[[OFFSET_RELOADED]]), !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_OVERFLOWED:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[OR_OV:.+]] = or i1 %[[COMPUTED_OFFSET_OVERFLOWED]], false, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint ptr %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], %[[COMPUTED_OFFSET]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne ptr %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[OR_OV]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_UGE_BASE:.*]] = icmp uge i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[COMPUTED_GEP_IS_UGE_BASE]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(ptr @[[LINE_1700]], i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(ptr @[[LINE_1700]], i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.ubsantrap(i8 19){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret ptr %[[ADD_PTR]]
#line 1700
  return base + offset;
}

#ifdef __cplusplus
}
#endif
