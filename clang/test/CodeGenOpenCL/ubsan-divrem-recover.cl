// OpenCL C v3.0 s6.5.1: integer division by zero and INT_MIN / -1 do not cause
// an exception but result in an unspecified value. The divrem check therefore
// reports and continues even when it would otherwise be fatal, and is not
// emitted as a trap. The add and the float division are controls.

// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s \
// RUN:   -fsanitize=integer-divide-by-zero,signed-integer-overflow -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefix=MIN --implicit-check-not=divrem_overflow_minimal_abort
// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=clc++2021 -disable-llvm-passes -emit-llvm -o - %s \
// RUN:   -fsanitize=integer-divide-by-zero,signed-integer-overflow -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefix=MIN --implicit-check-not=divrem_overflow_minimal_abort
// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s \
// RUN:   -fsanitize=integer-divide-by-zero,signed-integer-overflow -fsanitize-minimal-runtime \
// RUN:   -fsanitize-recover=integer-divide-by-zero,signed-integer-overflow \
// RUN:   | FileCheck %s --check-prefix=REC --implicit-check-not=minimal_abort
// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s \
// RUN:   -fsanitize=integer-divide-by-zero,signed-integer-overflow \
// RUN:   | FileCheck %s --check-prefix=FULL --implicit-check-not=divrem_overflow_abort
// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s \
// RUN:   -fsanitize=integer-divide-by-zero,signed-integer-overflow \
// RUN:   -fsanitize-trap=integer-divide-by-zero,signed-integer-overflow \
// RUN:   | FileCheck %s --check-prefix=TRAP --implicit-check-not=__ubsan_handle \
// RUN:     --implicit-check-not='ubsantrap(i8 3)'
// RUN: %clang_cc1 -triple spir64-unknown-unknown -cl-std=CL3.0 -disable-llvm-passes -emit-llvm -o - %s \
// RUN:   -fsanitize=float-divide-by-zero -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefix=FDIV --implicit-check-not='divrem_overflow_minimal()'
// RUN: %clang_cc1 -triple spir64-unknown-unknown -x c -disable-llvm-passes -emit-llvm -o - %s \
// RUN:   -fsanitize=integer-divide-by-zero,signed-integer-overflow -fsanitize-minimal-runtime \
// RUN:   | FileCheck %s --check-prefix=C --implicit-check-not='divrem_overflow_minimal()'

// MIN-LABEL:  define {{.*}}f_sdiv
// MIN:        call {{.*}}@__ubsan_handle_divrem_overflow_minimal()
// REC-LABEL:  define {{.*}}f_sdiv
// REC:        call {{.*}}@__ubsan_handle_divrem_overflow_minimal()
// FULL-LABEL: define {{.*}}f_sdiv
// FULL:       call {{.*}}@__ubsan_handle_divrem_overflow(
// C-LABEL:    define {{.*}}f_sdiv
// C:          call {{.*}}@__ubsan_handle_divrem_overflow_minimal_abort()
int f_sdiv(int a, int b) { return a / b; }

// MIN-LABEL:  define {{.*}}f_srem
// MIN:        call {{.*}}@__ubsan_handle_divrem_overflow_minimal()
// REC-LABEL:  define {{.*}}f_srem
// REC:        call {{.*}}@__ubsan_handle_divrem_overflow_minimal()
// FULL-LABEL: define {{.*}}f_srem
// FULL:       call {{.*}}@__ubsan_handle_divrem_overflow(
// C-LABEL:    define {{.*}}f_srem
// C:          call {{.*}}@__ubsan_handle_divrem_overflow_minimal_abort()
int f_srem(int a, int b) { return a % b; }

// MIN-LABEL:  define {{.*}}f_udiv
// MIN:        call {{.*}}@__ubsan_handle_divrem_overflow_minimal()
// REC-LABEL:  define {{.*}}f_udiv
// REC:        call {{.*}}@__ubsan_handle_divrem_overflow_minimal()
// FULL-LABEL: define {{.*}}f_udiv
// FULL:       call {{.*}}@__ubsan_handle_divrem_overflow(
// C-LABEL:    define {{.*}}f_udiv
// C:          call {{.*}}@__ubsan_handle_divrem_overflow_minimal_abort()
unsigned f_udiv(unsigned a, unsigned b) { return a / b; }

// MIN-LABEL:  define {{.*}}f_add
// MIN:        call {{.*}}@__ubsan_handle_add_overflow_minimal_abort()
// REC-LABEL:  define {{.*}}f_add
// REC:        call {{.*}}@__ubsan_handle_add_overflow_minimal()
// FULL-LABEL: define {{.*}}f_add
// FULL:       call {{.*}}@__ubsan_handle_add_overflow_abort(
// TRAP-LABEL: define {{.*}}f_add
// TRAP:       call void @llvm.ubsantrap(i8 0)
int f_add(int a, int b) { return a + b; }

// FDIV-LABEL: define {{.*}}f_fdiv
// FDIV:       call {{.*}}@__ubsan_handle_divrem_overflow_minimal_abort()
float f_fdiv(float a, float b) { return a / b; }
