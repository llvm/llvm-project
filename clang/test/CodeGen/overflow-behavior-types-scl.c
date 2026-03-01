// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu %t/test.c -fsanitize-ignorelist=%t/sio.scl \
// RUN: -fexperimental-overflow-behavior-types -fsanitize=signed-integer-overflow -emit-llvm -o - | FileCheck %s --check-prefix=SIO

// RUN: %clang_cc1 -triple x86_64-linux-gnu %t/test.c -fsanitize-ignorelist=%t/uio.scl \
// RUN: -fexperimental-overflow-behavior-types -fsanitize=unsigned-integer-overflow -emit-llvm -o - | FileCheck %s --check-prefix=UIO

// RUN: %clang_cc1 -triple x86_64-linux-gnu %t/test.c -fsanitize-ignorelist=%t/trunc.scl \
// RUN: -fexperimental-overflow-behavior-types -fsanitize=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation \
// RUN: -emit-llvm -o - | FileCheck %s --check-prefix=TRUNC

//--- sio.scl
[signed-integer-overflow]
# ignore signed-integer-overflow instrumentation across all types
type:*

//--- uio.scl
[unsigned-integer-overflow]
# ignore unsigned-integer-overflow instrumentation across all types
type:*

//--- trunc.scl
[{implicit-unsigned-integer-truncation,implicit-signed-integer-truncation}]
type:*

//--- test.c
#define __wrap __attribute__((overflow_behavior("wrap")))
#define __no_trap __attribute__((overflow_behavior("trap")))

// SIO-LABEL: define {{.*}} @foo
// UIO-LABEL: define {{.*}} @foo
void foo(void) {
  // SIO-LABEL: load volatile i32, ptr @a, align 4
  volatile extern int a;
  volatile extern char b;
  volatile extern char __ob_trap c; // nowrap has precedence over scl entries

  // SIO: add nsw i32
  (a + 1);
  // SIO: add nsw i32
  (b + 1);
  // SIO: @llvm.sadd.with.overflow.i32
  (c + 1);

  // UIO-LABEL: load volatile i32, ptr @d, align 4
  volatile extern unsigned int d;
  volatile extern unsigned short __ob_trap e;
  // UIO: add i32
  (d + 1);
  // UIO: @llvm.sadd.with.overflow.i32
  (e + 1);
}

// TRUNC-LABEL: define {{.*}} @bar
void bar(int value) {
  // TRUNC: %[[V0:.*]] = load i32, ptr %value.addr
  // TRUNC-NEXT: %[[CONV:.*]] = trunc i32 %[[V0]] to i8
  // TRUNC-NEXT: %[[ANYEXT:.*]] = zext i8 %[[CONV]] to i32
  // TRUNC-NEXT: %[[TCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[V0]]
  // TRUNC-NEXT: br i1 %[[TCHECK]], {{.*}}%handler.implicit_conversion
  unsigned char __ob_trap a = value;
}
