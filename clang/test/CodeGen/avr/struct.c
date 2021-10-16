// RUN: %clang_cc1 -triple avr -emit-llvm %s -o - | FileCheck %s

// Structure that is more than 8 bytes.
struct s10 {
  int a, b, c, d, e;
};

// Structure that is less than 8 bytes.
struct s06 {
  int a, b, c;
};

struct s10 foo10(int a, int b, int c) {
  struct s10 a0;
  return a0;
}

struct s06 foo06(int a, int b, int c) {
  struct s06 a0;
  return a0;
}

// CHECK: %struct.s10 = type { i16, i16, i16, i16, i16 }
// CHECK: %struct.s06 = type { i16, i16, i16 }
// CHECK: define{{.*}} void @foo10(%struct.s10* {{.*}}, i16 noundef %a, i16 noundef %b, i16 noundef %c)
// CHECK: define{{.*}} %struct.s06 @foo06(i16 noundef %a, i16 noundef %b, i16 noundef %c)
