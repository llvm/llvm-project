// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple x86_64-linux-gnu %t/test.c -fsanitize-ignorelist=%t/sio.scl -foverflow-behavior-types -fsanitize=signed-integer-overflow -emit-llvm -o - | FileCheck %s --check-prefix=SIO
// RUN: %clang_cc1 -triple x86_64-linux-gnu %t/test.c -fsanitize-ignorelist=%t/uio.scl -foverflow-behavior-types -fsanitize=unsigned-integer-overflow -emit-llvm -o - | FileCheck %s --check-prefix=UIO

//--- sio.scl
[signed-integer-overflow]
# ignore signed-integer-overflow instrumentation across all types
type:*

//--- uio.scl
[unsigned-integer-overflow]
# ignore unsigned-integer-overflow instrumentation across all types
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
