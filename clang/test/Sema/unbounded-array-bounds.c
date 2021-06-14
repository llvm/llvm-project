// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-X86-ADDR64 %s  \
// RUN:              --implicit-check-not 'past the last possible element'
// RUN: %clang_cc1 -triple i386-pc-linux-gnu   -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-I386-ADDR32 %s \
// RUN:              --implicit-check-not 'past the last possible element'
// RUN: %clang_cc1 -triple avr-pc-linux-gnu    -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-AVR-ADDR16 %s  \
// RUN:              --implicit-check-not 'past the last possible element'

struct S {
  long long a;
  char b;
  long long c;
  short d;
};

struct S s[];

void f1() {
  ++s[3].a;
  ++s[7073650413200313099].b;
  // CHECK-X86-ADDR64:  :[[@LINE-1]]:5: warning: array index 7073650413200313099 refers past the last possible element for an array in 64-bit address space containing 256-bit (32-byte) elements (max possible 576460752303423488 elements)
  // CHECK-I386-ADDR32: :[[@LINE-2]]:5: warning: {{.*}} past the last possible element {{.*}} in 32-bit {{.*}} (max possible 178956970 elements)
  // CHECK-AVR-ADDR16:  :[[@LINE-3]]:5: warning: {{.*}} past the last possible element {{.*}} in 16-bit {{.*}} (max possible 3276 elements)
  ++s[7073650].c;
  // CHECK-AVR-ADDR16:  :[[@LINE-1]]:5: warning: {{.*}} past the last possible element {{.*}} in 16-bit {{.*}} (max possible 3276 elements)
}

long long ll[];

void f2() {
  ++ll[3];
  ++ll[2705843009213693952];
  // CHECK-X86-ADDR64:  :[[@LINE-1]]:5: warning: {{.*}} past the last possible element {{.*}} in 64-bit {{.*}} (max possible 2305843009213693952 elements)
  // CHECK-I386-ADDR32: :[[@LINE-2]]:5: warning: {{.*}} past the last possible element {{.*}} in 32-bit {{.*}} (max possible 536870912 elements)
  // CHECK-AVR-ADDR16:  :[[@LINE-3]]:5: warning: {{.*}} past the last possible element {{.*}} in 16-bit {{.*}} (max possible 8192 elements)
  ++ll[847073650];
  // CHECK-I386-ADDR32: :[[@LINE-1]]:5: warning: {{.*}} past the last possible element {{.*}} in 32-bit {{.*}} (max possible 536870912 elements)
  // CHECK-AVR-ADDR16:  :[[@LINE-2]]:5: warning: {{.*}} past the last possible element {{.*}} in 16-bit {{.*}} (max possible 8192 elements)
}

void f3(struct S p[]) {
  ++p[3].a;
  ++p[7073650413200313099].b;
  // CHECK-X86-ADDR64:  :[[@LINE-1]]:5: warning: {{.*}} past the last possible element {{.*}} in 64-bit {{.*}} (max possible 576460752303423488 elements)
  // CHECK-I386-ADDR32: :[[@LINE-2]]:5: warning: {{.*}} past the last possible element {{.*}} in 32-bit {{.*}} (max possible 178956970 elements)
  // CHECK-AVR-ADDR16:  :[[@LINE-3]]:5: warning: {{.*}} past the last possible element {{.*}} in 16-bit {{.*}} (max possible 3276 elements)
  ++p[7073650].c;
  // CHECK-AVR-ADDR16:  :[[@LINE-1]]:5: warning: {{.*}} past the last possible element {{.*}} in 16-bit {{.*}} (max possible 3276 elements)
}

void f4(struct S *p) {
  p += 3;
  p += 7073650413200313099;
  // CHECK-X86-ADDR64:  :[[@LINE-1]]:3: warning: the pointer incremented by 7073650413200313099 refers past the last possible element for an array in 64-bit address space containing 256-bit (32-byte) elements (max possible 576460752303423488 elements)
  // CHECK-I386-ADDR32: :[[@LINE-2]]:3: warning: {{.*}} past the last possible element {{.*}} in 32-bit {{.*}} (max possible 178956970 elements)
  // CHECK-AVR-ADDR16:  :[[@LINE-3]]:3: warning: {{.*}} past the last possible element {{.*}} in 16-bit {{.*}} (max possible 3276 elements)
  p += 7073650;
  // CHECK-AVR-ADDR16:  :[[@LINE-1]]:3: warning: {{.*}} past the last possible element {{.*}} in 16-bit {{.*}} (max possible 3276 elements)
}

struct BQ {
  struct S bigblock[3276];
};

struct BQ bq[];

void f5() {
  ++bq[0].bigblock[0].a;
  ++bq[1].bigblock[0].a;
  // CHECK-AVR-ADDR16:  :[[@LINE-1]]:5: warning: {{.*}} past the last possible element {{.*}} in 16-bit {{.*}} (max possible 1 element)
}

void f6() {
  int ints[] = {1, 3, 5, 7, 8, 6, 4, 5, 9};
  int const n_ints = sizeof(ints) / sizeof(int);
  unsigned long long const N = 3;

  int *middle = &ints[0] + n_ints / 2;
  // Should NOT produce a warning.
  *(middle + 5 - N) = 22;
}
