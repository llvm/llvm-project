// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -I%S/Inputs -fclangir -fclangir-idiom-recognizer -fclangir-lib-opt -emit-cir -mmlir --mlir-print-ir-after=cir-lib-opt %s -o /dev/null 2>&1 | FileCheck %s
#include "string.h"

// Test strlen(str) ==|!=|> 0 --> *str ==|!=|> 0
int test_strlen_eq_zero(const char *str) {
// CHECK-LABEL:   cir.func{{.*}} @_Z19test_strlen_eq_zeroPKc(
// CHECK-SAME:      %[[ARG_STR:.*]]: !cir.ptr<!s8i>
// CHECK:           %[[VAR_STR:.*]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CHECK:           cir.store %[[ARG_STR]], %[[VAR_STR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CHECK:           %[[STR:.*]] = cir.load{{.*}} %[[VAR_STR]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
// CHECK:           %[[FIRST_CHAR:.*]] = cir.load{{.*}} %[[STR]] : !cir.ptr<!s8i>, !s8i
// CHECK:           %[[FIRST_INT:.*]] = cir.cast integral %[[FIRST_CHAR]] : !s8i -> !u64i
// CHECK:           %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
// CHECK:           %[[CMP:.*]] = cir.cmp(eq, %[[FIRST_INT]], %[[ZERO]]) : !u64i, !cir.bool
// CHECK:         }

    return strlen(str) == 0ULL; // expected-remark "strlen opt: transformed strlen into load"
}

// Test strlen(str) <|>= len --> memchr(str, 0, len) <|>= len
int test_strlen_lt_var(const char *str, size_t len) {
  // CHECK-LABEL:   cir.func{{.*}} @_Z18test_strlen_lt_varPKcm(
  // CHECK:           %[[STR:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
  // CHECK:           %[[LEN:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!u64i>, !u64i
  // CHECK:           %[[STR_VOID_PTR:.*]] = cir.cast bitcast %[[STR]] : !cir.ptr<!s8i> -> !cir.ptr<!void>
  // CHECK:           %[[NULL_CHAR:.*]] = cir.const #cir.int<0> : !s8i
  // CHECK:           %[[NULL_INT:.*]] = cir.cast integral %[[NULL_CHAR]] : !s8i -> !s32i
  // CHECK:           %[[FOUND_VOID_PTR:.*]] = cir.libc.memchr(%[[STR_VOID_PTR]], %[[NULL_INT]], %[[LEN]])
  // CHECK:           %[[FOUND_STR_PTR:.*]] = cir.cast bitcast %[[FOUND_VOID_PTR]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
  // CHECK:           %[[DIST:.*]] = cir.ptr_diff %[[FOUND_STR_PTR]], %[[STR]] : !cir.ptr<!s8i> -> !u64i
  // CHECK:           %[[CMP:.*]] = cir.cmp(lt, %[[DIST]], %[[LEN]]) : !u64i, !cir.bool
  // CHECK:         }

    return strlen(str) < len; // expected-remark "strlen opt: transformed strlen into memchr"
}

// Test strlen(str) >|<=|==|!= len --> memchr(str, 0, len + 1) >|<=|==|!= len
int test_strlen_eq_var(const char *str, size_t len) {
  // CHECK-LABEL:   cir.func{{.*}} @_Z18test_strlen_eq_varPKcm(
  // CHECK:           %[[ONE:.*]] = cir.const #cir.int<1>
  // CHECK:           %[[LEN_PLUS_ONE:.*]] = cir.binop(add, %{{.*}}, %[[ONE]])
  // CHECK:           %[[RESULT:.*]] = cir.libc.memchr(%{{.*}}, %{{.*}}, %[[LEN_PLUS_ONE]])
  // CHECK:         }

    return strlen(str) == len; // expected-remark "strlen opt: transformed strlen into memchr"
}

// Applicability tests:

// Multiple users, not applicable.
int test_strlen_multiple_users(const char *str, size_t len1, size_t len2) {
  // Check that we still have a strlen op.
  // CHECK-LABEL:   cir.func{{.*}} @_Z26test_strlen_multiple_usersPKcmm(
  // CHECK:           %[[LEN:.*]] = {{.*}}strlen(
  // CHECK:         }

  size_t len = strlen(str);
  return len1 < len && len < len2; // expected-remark "strlen opt: result of strlen has more than one use"
}

// Non-comparison user, not applicable.
int test_strlen_non_cmp_users(const char *str) {
  // Check that we still have a strlen op.
  // CHECK-LABEL:   cir.func{{.*}} @_Z25test_strlen_non_cmp_usersPKc(
  // CHECK:           %[[LEN:.*]] = {{.*}}strlen(
  // CHECK:         }

  return strlen(str); // expected-remark "strlen opt: could not find cir.cmp user of strlen result"
}

// Memory operation blocks move.
int test_strlen_store_between_def_and_use(const char *str, size_t *ptr) {
  // Check that we still have either a strlen op, or a call to strlen.
  // CHECK-LABEL:   cir.func{{.*}} @_Z37test_strlen_store_between_def_and_usePKcPm(
  // CHECK:           %[[LEN:.*]] = {{.*}}strlen
  // CHECK:         }

  size_t len = strlen(str);
  *ptr = 10;
  return len < *ptr; // expected-remark "strlen opt: could not move max length before strlen"
}

// Can't adjust value being compared.
int test_strlen_cant_adjust(const char *str) {
  // Check that we still have either a strlen op, or a call to strlen.
  // CHECK-LABEL:   cir.func{{.*}} @_Z23test_strlen_cant_adjustPKc(
  // CHECK:           %[[LEN:.*]] = {{.*}}strlen
  // CHECK:         }

  return strlen(str) < 10.0; // expected-remark "strlen opt: could not adjust the max value"
}
