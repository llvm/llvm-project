// signed-integer-overflow is a group of signed-integer-overflow-divide, which
// checks INT_MIN / -1 and INT_MIN % -1, and signed-integer-overflow-arith,
// which checks every other signed overflow.

// RUN: rm -rf %t && split-file %s %t

// cc1 still accepts signed-integer-overflow, which was a single sanitizer.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow | FileCheck %t/test.c --check-prefixes=CHECK,ABS,ARITH,DIVIDE,AFATAL,DFATAL
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow-arith,signed-integer-overflow-divide | FileCheck %t/test.c --check-prefixes=CHECK,ABS,ARITH,DIVIDE,AFATAL,DFATAL
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow-arith | FileCheck %t/test.c --check-prefixes=CHECK,ABS,ARITH,NO-DIVIDE,AFATAL
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow-divide | FileCheck %t/test.c --check-prefixes=CHECK,NO-ARITH,DIVIDE,DFATAL,NO-ABS
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow -fsanitize-recover=signed-integer-overflow | FileCheck %t/test.c --check-prefixes=CHECK,ABS,ARITH,DIVIDE,RECOVER
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow | FileCheck %t/test.c --check-prefixes=CHECK,TRAP
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow -fsanitize-recover=signed-integer-overflow-divide -fsanitize-trap=signed-integer-overflow-arith | FileCheck %t/test.c --check-prefixes=CHECK,MIXED
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow -fsanitize-ignorelist=%t/group.ignorelist | FileCheck %t/test.c --check-prefixes=CHECK,ABS,NO-ARITH,NO-DIVIDE
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow -fsanitize-ignorelist=%t/divide.ignorelist | FileCheck %t/test.c --check-prefixes=CHECK,ABS,ARITH,NO-DIVIDE,AFATAL
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow -fsanitize-skip-hot-cutoff=signed-integer-overflow=1.0,signed-integer-overflow-divide=0.0 | FileCheck %t/test.c --check-prefixes=CHECK,DIVIDE,DFATAL,CUTOFF
// RUN: %clang_cc1 -triple x86_64-linux-gnu -debug-info-kind=limited -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow -fsanitize-annotate-debug-info=signed-integer-overflow | FileCheck %t/test.c --check-prefixes=CHECK,ABS,ARITH,DIVIDE,ANNOTATE
// RUN: %clang_cc1 -triple x86_64-linux-gnu -ftrapv -emit-llvm -o - %t/test.c -fsanitize=signed-integer-overflow-divide | FileCheck %t/test.c --check-prefixes=TRAPV,DIVIDE,DFATAL

//--- group.ignorelist
[signed-integer-overflow]
type:int

//--- divide.ignorelist
[signed-integer-overflow-divide]
type:int

//--- test.c
// CHECK-LABEL: define {{.*}} @div(
// TRAPV-LABEL: define {{.*}} @div(
// DIVIDE:       icmp ne i32 %{{.*}}, -2147483648
// DFATAL:       call void @__ubsan_handle_divrem_overflow_abort(
// RECOVER:      call void @__ubsan_handle_divrem_overflow(
// TRAP:         call void @llvm.ubsantrap(i8 3)
// MIXED:        call void @__ubsan_handle_divrem_overflow(
// NO-DIVIDE-NOT: __ubsan_handle_divrem_overflow
// CHECK:        sdiv i32
// TRAPV:        sdiv i32
int div(int x, int y) { return x / y; }

// CHECK-LABEL: define {{.*}} @rem(
// TRAPV-LABEL: define {{.*}} @rem(
// DIVIDE:       icmp ne i32 %{{.*}}, -2147483648
// DFATAL:       call void @__ubsan_handle_divrem_overflow_abort(
// RECOVER:      call void @__ubsan_handle_divrem_overflow(
// TRAP:         call void @llvm.ubsantrap(i8 3)
// MIXED:        call void @__ubsan_handle_divrem_overflow(
// NO-DIVIDE-NOT: __ubsan_handle_divrem_overflow
// CHECK:        srem i32
// TRAPV:        srem i32
int rem(int x, int y) { return x % y; }

// CHECK-LABEL: define {{.*}} @add(
// TRAPV-LABEL: define {{.*}} @add(
// ARITH:        call { i32, i1 } @llvm.sadd.with.overflow.i32(
// AFATAL:       call void @__ubsan_handle_add_overflow_abort(
// RECOVER:      call void @__ubsan_handle_add_overflow(
// TRAP:         call void @llvm.ubsantrap(i8 0)
// MIXED:        call void @llvm.ubsantrap(i8 0)
// TRAPV:        call void @llvm.ubsantrap(i8 0)
// NO-ARITH:     add nsw i32
// NO-ARITH-NOT: __ubsan_handle
// CUTOFF-NOT:   __ubsan_handle
// CHECK:        ret i32
int add(int x, int y) { return x + y; }

// CHECK-LABEL: define {{.*}} @neg(
// TRAPV-LABEL: define {{.*}} @neg(
// ARITH:        call { i32, i1 } @llvm.ssub.with.overflow.i32(
// AFATAL:       call void @__ubsan_handle_negate_overflow_abort(
// TRAPV:        call void @llvm.ubsantrap(i8 21)
// NO-ARITH:     sub nsw i32 0,
// NO-ARITH-NOT: __ubsan_handle
// CUTOFF-NOT:   __ubsan_handle
// CHECK:        ret i32
int neg(int x) { return -x; }

// Type ignorelists do not apply to __builtin_abs.
// CHECK-LABEL: define {{.*}} @abs_(
// TRAPV-LABEL: define {{.*}} @abs_(
// ABS:          call { i32, i1 } @llvm.ssub.with.overflow.i32(
// AFATAL:       call void @__ubsan_handle_negate_overflow_abort(
// ANNOTATE:     call void @__ubsan_handle_negate_overflow_abort({{.*}} !dbg
// TRAPV:        call void @llvm.ubsantrap(i8 21)
// NO-ABS:       call i32 @llvm.abs.i32(i32 %{{.*}}, i1 true)
// NO-ABS-NOT:   __ubsan_handle
// CUTOFF-NOT:   __ubsan_handle
// CHECK:        ret i32
int abs_(int x) { return __builtin_abs(x); }

// The group in no_sanitize turns off both leaves.
// CHECK-LABEL: define {{.*}} @div_no_sanitize_group(
// TRAPV-LABEL: define {{.*}} @div_no_sanitize_group(
// CHECK-NOT:    __ubsan_handle
// CHECK-NOT:    ubsantrap
// TRAPV-NOT:    __ubsan_handle
// CHECK:        ret i32
int div_no_sanitize_group(int x, int y) __attribute__((no_sanitize("signed-integer-overflow")));
__attribute__((no_sanitize("signed-integer-overflow")))
int div_no_sanitize_group(int x, int y) { return x / y + x * y; }

// A leaf in no_sanitize turns off just that leaf.
// CHECK-LABEL: define {{.*}} @div_no_sanitize_divide(
// TRAPV-LABEL: define {{.*}} @div_no_sanitize_divide(
// NO-DIVIDE-NOT: __ubsan_handle_divrem_overflow
// DIVIDE-NOT:    __ubsan_handle_divrem_overflow
// CHECK:         sdiv i32
// TRAPV:         sdiv i32
// ARITH:         @llvm.smul.with.overflow.i32(
// CHECK:         ret i32
__attribute__((no_sanitize("signed-integer-overflow-divide")))
int div_no_sanitize_divide(int x, int y) { return x / y + x * y; }
