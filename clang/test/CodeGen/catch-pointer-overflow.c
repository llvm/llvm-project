// RUN: %clang_cc1 -x c -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-NOSANITIZE
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-C,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-C,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-C,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// RUN: %clang_cc1 -x c++ -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-NOSANITIZE
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fno-sanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-CPP,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-CPP,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-CPP,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_100:.*]] = {{.*}}, i32 100, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_200:.*]] = {{.*}}, i32 200, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_300:.*]] = {{.*}}, i32 300, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_400:.*]] = {{.*}}, i32 400, i32 15 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_500:.*]] = {{.*}}, i32 500, i32 7 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_600:.*]] = {{.*}}, i32 600, i32 7 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_700:.*]] = {{.*}}, i32 700, i32 3 } }
// CHECK-SANITIZE-ANYRECOVER-DAG: @[[LINE_800:.*]] = {{.*}}, i32 800, i32 3 } }

#ifdef __cplusplus
extern "C" {
#endif

char *add_unsigned(char *base, unsigned long offset) {
  // CHECK:                           define i8* @add_unsigned(i8* %[[BASE:.*]], i64 %[[OFFSET:.*]])
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
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR]]
#line 100
  return base + offset;
}

char *sub_unsigned(char *base, unsigned long offset) {
  // CHECK:                           define i8* @sub_unsigned(i8* %[[BASE:.*]], i64 %[[OFFSET:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca i8*, align 8
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store i8* %[[BASE]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[IDX_NEG:.*]] = sub i64 0, %[[OFFSET_RELOADED]]
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BASE_RELOADED]], i64 %[[IDX_NEG]]
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_AGGREGATE:.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %[[IDX_NEG]]), !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_OVERFLOWED:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint i8* %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], %[[COMPUTED_OFFSET]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne i8* %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:             %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[COMPUTED_OFFSET_OVERFLOWED]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_ULE_BASE:.*]] = icmp ule i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[COMPUTED_GEP_IS_ULE_BASE]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_200]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_200]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR]]
#line 200
  return base - offset;
}

char *add_signed(char *base, signed long offset) {
  // CHECK:                           define i8* @add_signed(i8* %[[BASE:.*]], i64 %[[OFFSET:.*]])
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
  // CHECK-SANITIZE-NEXT:               %[[POSORZEROVALID:.*]] = icmp uge i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[POSORZEROOFFSET:.*]] = icmp sge i64 %[[COMPUTED_OFFSET]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[NEGVALID:.*]] = icmp ult i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[APPLYING_OFFSET_DID_NOT_OVERFLOW:.*]] = select i1 %[[POSORZEROOFFSET]], i1 %[[POSORZEROVALID]], i1 %[[NEGVALID]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[APPLYING_OFFSET_DID_NOT_OVERFLOW]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_300]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_300]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR]]
#line 300
  return base + offset;
}

char *sub_signed(char *base, signed long offset) {
  // CHECK:                           define i8* @sub_signed(i8* %[[BASE:.*]], i64 %[[OFFSET:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca i8*, align 8
  // CHECK-NEXT:                        %[[OFFSET_ADDR:.*]] = alloca i64, align 8
  // CHECK-NEXT:                        store i8* %[[BASE]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        store i64 %[[OFFSET]], i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[OFFSET_RELOADED:.*]] = load i64, i64* %[[OFFSET_ADDR]], align 8
  // CHECK-NEXT:                        %[[IDX_NEG:.*]] = sub i64 0, %[[OFFSET_RELOADED]]
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BASE_RELOADED]], i64 %[[IDX_NEG]]
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_AGGREGATE:.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %[[IDX_NEG]]), !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_OVERFLOWED:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET:.*]] = extractvalue { i64, i1 } %[[COMPUTED_OFFSET_AGGREGATE]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint i8* %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], %[[COMPUTED_OFFSET]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne i8* %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:             %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW:.*]] = xor i1 %[[COMPUTED_OFFSET_OVERFLOWED]], true, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[POSORZEROVALID:.*]] = icmp uge i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[POSORZEROOFFSET:.*]] = icmp sge i64 %[[COMPUTED_OFFSET]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[NEGVALID:.*]] = icmp ult i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[APPLYING_OFFSET_DID_NOT_OVERFLOW:.*]] = select i1 %[[POSORZEROOFFSET]], i1 %[[POSORZEROVALID]], i1 %[[NEGVALID]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_DID_NOT_OVERFLOW:.*]] = and i1 %[[APPLYING_OFFSET_DID_NOT_OVERFLOW]], %[[COMPUTED_OFFSET_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[GEP_DID_NOT_OVERFLOW]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_400]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_400]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR]]
#line 400
  return base - offset;
}

char *postinc(char *base) {
  // CHECK:                           define i8* @postinc(i8* %[[BASE:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca i8*, align 8
  // CHECK-NEXT:                        store i8* %[[BASE]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BASE_RELOADED]], i32 1
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
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_500]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_500]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        store i8* %[[ADD_PTR]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR_RELOADED]]
#line 500
  base++;
  return base;
}

char *postdec(char *base) {
  // CHECK:                           define i8* @postdec(i8* %[[BASE:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca i8*, align 8
  // CHECK-NEXT:                        store i8* %[[BASE]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BASE_RELOADED]], i32 -1
  // CHECK-SANITIZE-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint i8* %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], -1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne i8* %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:             %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_ULE_BASE:.*]] = icmp ule i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[COMPUTED_GEP_IS_ULE_BASE]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_600]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_600]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        store i8* %[[ADD_PTR]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR_RELOADED]]
#line 600
  base--;
  return base;
}

char *preinc(char *base) {
  // CHECK:                           define i8* @preinc(i8* %[[BASE:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca i8*, align 8
  // CHECK-NEXT:                        store i8* %[[BASE]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BASE_RELOADED]], i32 1
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
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_700]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_700]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        store i8* %[[ADD_PTR]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR_RELOADED]]
#line 700
  ++base;
  return base;
}

char *predec(char *base) {
  // CHECK:                           define i8* @predec(i8* %[[BASE:.*]])
  // CHECK-NEXT:                      [[ENTRY:.*]]:
  // CHECK-NEXT:                        %[[BASE_ADDR:.*]] = alloca i8*, align 8
  // CHECK-NEXT:                        store i8* %[[BASE]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[BASE_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR:.*]] = getelementptr inbounds i8, i8* %[[BASE_RELOADED]], i32 -1
  // CHECK-SANITIZE-NEXT:               %[[BASE_RELOADED_INT:.*]] = ptrtoint i8* %[[BASE_RELOADED]] to i64, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP:.*]] = add i64 %[[BASE_RELOADED_INT]], -1, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[BASE_IS_NOT_NULLPTR:.*]] = icmp ne i8* %[[BASE_RELOADED]], null, !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_NOT_NULL:.*]] = icmp ne i64 %[[COMPUTED_GEP]], 0, !nosanitize
  // CHECK-SANITIZE-C-NEXT:             %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = and i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-CPP-NEXT:           %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL:.*]] = icmp eq i1 %[[BASE_IS_NOT_NULLPTR]], %[[COMPUTED_GEP_IS_NOT_NULL]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[COMPUTED_GEP_IS_ULE_BASE:.*]] = icmp ule i64 %[[COMPUTED_GEP]], %[[BASE_RELOADED_INT]], !nosanitize
  // CHECK-SANITIZE-NEXT:               %[[GEP_IS_OKAY:.*]] = and i1 %[[BOTH_POINTERS_ARE_NULL_OR_BOTH_ARE_NONNULL]], %[[COMPUTED_GEP_IS_ULE_BASE]], !nosanitize
  // CHECK-SANITIZE-NEXT:               br i1 %[[GEP_IS_OKAY]], label %[[CONT:.*]], label %[[HANDLER_POINTER_OVERFLOW:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE:                  [[HANDLER_POINTER_OVERFLOW]]:
  // CHECK-SANITIZE-NORECOVER-NEXT:     call void @__ubsan_handle_pointer_overflow_abort(i8* bitcast ({ {{{.*}}} }* @[[LINE_800]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-RECOVER-NEXT:       call void @__ubsan_handle_pointer_overflow(i8* bitcast ({ {{{.*}}} }* @[[LINE_800]] to i8*), i64 %[[BASE_RELOADED_INT]], i64 %[[COMPUTED_GEP]])
  // CHECK-SANITIZE-TRAP-NEXT:          call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT:   unreachable, !nosanitize
  // CHECK-SANITIZE:                  [[CONT]]:
  // CHECK-NEXT:                        store i8* %[[ADD_PTR]], i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        %[[ADD_PTR_RELOADED:.*]] = load i8*, i8** %[[BASE_ADDR]], align 8
  // CHECK-NEXT:                        ret i8* %[[ADD_PTR_RELOADED]]
#line 800
  --base;
  return base;
}

#ifdef __cplusplus
}
#endif
