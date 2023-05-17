// RUN: %clang_cc1 -triple avr -target-cpu atmega328 -emit-llvm %s -o - \
// RUN:     | FileCheck %s --check-prefix=AVR
// RUN: %clang_cc1 -triple avr -target-cpu attiny40 -emit-llvm %s -o - \
// RUN:     | FileCheck %s --check-prefix=TINY

// Structure that is more than 8 bytes.
struct s10 {
  int a, b, c, d, e;
};

// Structure that is less than 8 bytes but more than 4 bytes.
struct s06 {
  int a, b, c;
};

// Structure that is less than 4 bytes.
struct s04 {
  int a, b;
};

struct s10 foo10(int a, int b, int c) {
  struct s10 a0;
  return a0;
}

struct s06 foo06(int a, int b, int c) {
  struct s06 a0;
  return a0;
}

struct s04 foo04(int a, int b) {
  struct s04 a0;
  return a0;
}

long long fooi64(void) {
  return 0xaa5533;
}

// AVR: %struct.s10 = type { i16, i16, i16, i16, i16 }
// AVR: %struct.s06 = type { i16, i16, i16 }
// AVR: %struct.s04 = type { i16, i16 }
// AVR: define{{.*}} void @foo10(ptr {{.*}}, i16 noundef %a, i16 noundef %b, i16 noundef %c)
// AVR: define{{.*}} %struct.s06 @foo06(i16 noundef %a, i16 noundef %b, i16 noundef %c)
// AVR: define{{.*}} %struct.s04 @foo04(i16 noundef %a, i16 noundef %b)
// AVR: define{{.*}} i64 @fooi64()

// TINY: %struct.s10 = type { i16, i16, i16, i16, i16 }
// TINY: %struct.s06 = type { i16, i16, i16 }
// TINY: %struct.s04 = type { i16, i16 }
// TINY: define{{.*}} void @foo10(ptr {{.*}}, i16 noundef %a, i16 noundef %b, i16 noundef %c)
// TINY: define{{.*}} void @foo06(ptr {{.*}}, i16 noundef %a, i16 noundef %b, i16 noundef %c)
// TINY: define{{.*}} %struct.s04 @foo04(i16 noundef %a, i16 noundef %b)
// TINY: define{{.*}} void @fooi64(ptr {{.*}})
