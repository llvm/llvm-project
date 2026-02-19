// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types %s \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types %s -ftrapv \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types %s \
// RUN: -ftrapv -ftrapv-handler OVERFLOW_HANDLER \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=TRAPV-HANDLER

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types %s -fwrapv \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=DEFAULT

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types %s \
// RUN: -fsanitize-undefined-ignore-overflow-pattern=all \
// RUN: -fsanitize=signed-integer-overflow,unsigned-integer-overflow,implicit-signed-integer-truncation,implicit-unsigned-integer-truncation \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=EXCL

// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types %s \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=NOSAN

#define __wrap __attribute__((overflow_behavior("wrap")))
#define __no_trap __attribute__((overflow_behavior("trap")))

// DEFAULT-LABEL: define {{.*}} @test1
// TRAPV-HANDLER-LABEL: define {{.*}} @test1
// NOSAN-LABEL: define {{.*}} @test1
void test1(int __ob_wrap a, int __ob_trap b) {
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
void test2(unsigned char __ob_wrap a, unsigned char __ob_trap b) {
  // DEFAULT: add i32
  (a + 1);
  // DEFAULT: llvm.sadd.with.overflow.i32
  (b + 1);

  // DEFAULT: %[[T0:.*]] = load volatile i64, ptr @big
  // DEFAULT: trunc i64
  // DEFAULT-NOT: icmp eq i64
  // No truncation check for __ob_wrap destination
  volatile extern unsigned long long big;
  a = big;

  // DEFAULT: %[[T1:.*]] = load volatile i64, ptr @big
  // DEFAULT: %[[TRUNC2:.*]] = icmp eq i64 {{.*}} %[[T1]]
  // DEFAULT: br i1 %[[TRUNC2]], {{.*}} %handler.implicit_conversion
  b = big;
}

// DEFAULT-LABEL: define {{.*}} @test3
void test3(void) {
  volatile extern char __ob_wrap a;
  volatile extern short __ob_wrap b;
  // DEFAULT: add i32
  (a + b);

  // no_trap has precedence over wrap, regardless of bit widths
  volatile extern unsigned long long __ob_wrap c;
  volatile extern char __ob_trap d;

  // DEFAULT: %[[T0:.*]] = load volatile i64, ptr @c
  // DEFAULT: %[[T1:.*]] = load volatile i8, ptr @d
  // DEFAULT: @llvm.uadd.with.overflow.i64
  (c + d);

  volatile extern int __ob_trap e;
  volatile extern unsigned int __ob_wrap f;

  // DEFAULT: @llvm.usub.with.overflow.i32
  (e - f);
}

typedef int __attribute__((overflow_behavior(wrap))) wrap_int;
typedef int __attribute__((overflow_behavior(trap))) no_trap_int;
// DEFAULT-LABEL: define {{.*}} @typedefs
void typedefs(no_trap_int a, wrap_int b) {
  // DEFAULT: llvm.sadd.with.overflow.i32
  (a + 100);

  // DEFAULT: add i32
  (b + 100);
}

// EXCL-LABEL: define {{.*}} @ignored_patterns
void ignored_patterns(unsigned long __attribute__((overflow_behavior(trap))) a) {
  // EXCL: %[[T0:.*]] = load i64, ptr %a.addr
  // EXCL-NEXT: add i64 %[[T0]], -1
  while (a--) { /*...*/ }

  // EXCL: %[[T1:.*]] = load i64, ptr %a.addr
  // EXCL: %[[T2:.*]] = load volatile i64, ptr %b
  // EXCL-NEXT: add i64 %[[T1]], %[[T2]]
  volatile unsigned long __attribute__((overflow_behavior(trap))) b;
  if (a + b < a) { /*...*/ }
}

// NOSAN-LABEL: define {{.*}} @implicit_truncation_return
int implicit_truncation_return(__ob_trap unsigned long long result) {
  // NOSAN: trunc i64 {{.*}} to i32
  // NOSAN: sext i32 {{.*}} to i64
  // NOSAN: icmp eq i64
  // NOSAN: br i1 {{.*}}, label %{{.*}}, label %trap
  // NOSAN: trap:
  // NOSAN: call void @llvm.ubsantrap(i8 7)
  return result;
}

// NOSAN-LABEL: define {{.*}} @implicit_truncation_assignment
void implicit_truncation_assignment(__ob_trap unsigned long long result) {
  // NOSAN: trunc i64 {{.*}} to i32
  // NOSAN: sext i32 {{.*}} to i64
  // NOSAN: icmp eq i64
  // NOSAN: br i1 {{.*}}, label %{{.*}}, label %trap
  // NOSAN: trap:
  // NOSAN: call void @llvm.ubsantrap(i8 7)
  int a = result;
}

// NOSAN-LABEL: define {{.*}} @explicit_truncation_cast
int explicit_truncation_cast(__ob_trap unsigned long long result) {
  // NOSAN: trunc i64 {{.*}} to i32
  // NOSAN: sext i32 {{.*}} to i64
  // NOSAN: icmp eq i64
  // NOSAN: br i1 {{.*}}, label %{{.*}}, label %trap
  // NOSAN: trap:
  // NOSAN: call void @llvm.ubsantrap(i8 7)
  return (int)result;
}
