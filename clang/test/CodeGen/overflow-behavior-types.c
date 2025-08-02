// RUN: %clang_cc1 -triple x86_64-linux-gnu -foverflow-behavior-types %s \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT

// RUN: %clang_cc1 -triple x86_64-linux-gnu -foverflow-behavior-types %s -ftrapv \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT

// RUN: %clang_cc1 -triple x86_64-linux-gnu -foverflow-behavior-types %s \
// RUN: -ftrapv -ftrapv-handler OVERFLOW_HANDLER \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=TRAPV-HANDLER

// RUN: %clang_cc1 -triple x86_64-linux-gnu -foverflow-behavior-types %s -fwrapv \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT

// RUN: %clang_cc1 -triple x86_64-linux-gnu -foverflow-behavior-types %s \
// RUN: -fsanitize-undefined-ignore-overflow-pattern=all \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=EXCL

// RUN: %clang_cc1 -triple x86_64-linux-gnu -foverflow-behavior-types %s \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=NOSAN

#define __wrap __attribute__((overflow_behavior("wrap")))
#define __nowrap __attribute__((overflow_behavior("no_wrap")))

// DEFAULT-LABEL: define {{.*}} @test1
// TRAPV-HANDLER-LABEL: define {{.*}} @test1
// NOSAN-LABEL: define {{.*}} @test1
void test1(int __wrap a, int __nowrap b) {
  // DEFAULT: add i32
  // TRAPV-HANDLER: add i32
  // NOSAN: add i32
  (a + 1);

  // DEFAULT: llvm.sadd.with.overflow.i32
  // TRAPV-HANDLER: %[[T0:.*]] = load i32, ptr %b
  // TRAPV-HANDLER: call {{.*}} @OVERFLOW_HANDLER(i64 %[[T0]]
  // NOSAN: %[[T0:.*]] = load i32, ptr %b
  // NOSAN-NEXT: %[[T1:.*]] = call {{.*}} @llvm.sadd.with.overflow.i32(i32 %[[T0]]
  // NOSAN: %[[OF:.*]] = extractvalue {{.*}} %[[T1]], 1
  // NOSAN-NEXT: %[[XOR:.*]] = xor i1 %[[OF]]
  // NOSAN-NEXT: br i1 %[[XOR]]{{.*}}cont, label %[[TRAP:.*]], !prof
  // NOSAN: [[TRAP]]:
  // NOSAN-NEXT: call void @llvm.ubsantrap
  (b + 1);

  // DEFAULT: sub i32 0
  (-a);

  // DEFAULT: llvm.ssub.with.overflow.i32
  (-b);

  // DEFAULT: add i32
  a++;
  // DEFAULT: llvm.sadd.with.overflow.i32
  b++;

  // DEFAULT: add i32
  ++a;
  // DEFAULT: llvm.sadd.with.overflow.i32
  ++b;

  volatile extern int divisor;
  // DEFAULT: %[[T0:.*]] = load i32, ptr %a
  // DEFAULT-NEXT: %[[T1:.*]] = load volatile i32, ptr @divisor
  // DEFAULT-NOT: br {{.*}} %handler.divrem_overflow
  // DEFAULT: sdiv i32 %[[T0]], %[[T1]]
  a/divisor;

  // DEFAULT: %[[T0:.*]] = load i32, ptr %b
  // DEFAULT-NEXT: %[[T1:.*]] = load volatile i32, ptr @divisor
  // DEFAULT: br {{.*}} %handler.divrem_overflow
  b/divisor;
}

// DEFAULT-LABEL: define {{.*}} @test2
void test2(unsigned char __wrap a, unsigned char __nowrap b) {
  // DEFAULT: add i8
  (a + 1);
  // DEFAULT: llvm.uadd.with.overflow.i8
  (b + 1);

  // DEFAULT: %[[T0:.*]] = load volatile i64, ptr @big
  // DEFAULT: %[[TRUNC1:.*]] = icmp eq i64 {{.*}} %[[T0]]
  // DEFAULT-NOT: br i1 %[[TRUNC1]], {{.*}} %handler.implicit_conversion
  volatile extern unsigned long long big;
  a = big;

  // DEFAULT: %[[T1:.*]] = load volatile i64, ptr @big
  // DEFAULT: %[[TRUNC2:.*]] = icmp eq i64 {{.*}} %[[T1]]
  // DEFAULT: br i1 %[[TRUNC2]], {{.*}} %handler.implicit_conversion
  b = big;
}

// DEFAULT-LABEL: define {{.*}} @test3
void test3(void) {
  volatile extern char __wrap a;
  volatile extern short __wrap b;
  // less-than-int arithmetic is possible when one or more wrapping types are
  // present. When both operands are wrapping types, the larger of the two
  // types should be used as the result of the arithmetic.

  // DEFAULT: add i16
  (a + b);

  // nowrap has precedence over wrap, regardless of bit widths
  volatile extern unsigned long long __wrap c;
  volatile extern char __nowrap d;

  // DEFAULT: %[[T0:.*]] = load volatile i64, ptr @c
  // DEFAULT: %[[TRUNC1:.*]] = icmp eq i64 {{.*}} %[[T0]]
  // DEFAULT: br i1 %[[TRUNC1]]
  // DEFAULT: %[[T1:.*]] = load volatile i8, ptr @d
  // DEFAULT-NEXT: @llvm.sadd.with.overflow.i8
  (c + d);

  volatile extern int __nowrap e;
  volatile extern unsigned int __wrap f;

  // DEFAULT: @llvm.ssub.with.overflow.i32
  (e - f);
}

typedef int __attribute__((overflow_behavior(wrap))) wrap_int;
typedef int __attribute__((overflow_behavior(no_wrap))) nowrap_int;
// DEFAULT-LABEL: define {{.*}} @typedefs
void typedefs(nowrap_int a, wrap_int b) {
  // DEFAULT: llvm.sadd.with.overflow.i32
  (a + 100);

  // DEFAULT: add i32
  (b + 100);
}

// EXCL-LABEL: define {{.*}} @ignored_patterns
void ignored_patterns(unsigned long __attribute__((overflow_behavior(no_wrap))) a) {
  // EXCL: %[[T0:.*]] = load i64, ptr %a.addr
  // EXCL-NEXT: add i64 %[[T0]], -1
  while (a--) { /*...*/ }

  // EXCL: %[[T1:.*]] = load i64, ptr %a.addr
  // EXCL: %[[T2:.*]] = load volatile i64, ptr %b
  // EXCL-NEXT: add i64 %[[T1]], %[[T2]]
  volatile unsigned long __attribute__((overflow_behavior(no_wrap))) b;
  if (a + b < a) { /*...*/ }
}
