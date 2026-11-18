// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

#define CASES(n) case (n): case (n + 1): case (n + 2): case (n + 3):
#define CASES16(n) CASES(n) CASES(n + 4) CASES(n + 8) CASES(n + 12)
#define CASES64(n) CASES16(n) CASES16(n + 16) CASES16(n + 32) CASES16(n + 48)
#define CASES256(n) CASES64(n) CASES64(n + 64) CASES64(n + 128) CASES64(n + 192)
#define CASES1024(n) CASES256(n) CASES256(n + 256) CASES256(n + 512) CASES256(n + 768)
#define CASES4192(n) CASES1024(n) CASES1024(n + 1024) CASES1024(n + 2048) CASES1024(n + 3072)
#define CASES16768(n) CASES4192(n) CASES4192(n + 4192) CASES4192(n + 8384) CASES4192(n + 12576)
#define CASES_STARTING_AT(n) CASES16768(n) CASES16768(n + 16768) CASES16768(n + 33536) CASES16768(n + 50304)

// Check this doesn't cause the compiler to crash
void foo() {
  // CHECK-LABEL: @foo
  // CHECK-NOT: switch{{ }}
  // CHECK-NOT: br{{ }}

  // 37 does not match a switch case
  switch (37) {
    CASES_STARTING_AT(100)
    break;
  }

  // 2000 matches a switch case
  switch(2000) {
    CASES_STARTING_AT(0)
    break;
  }
}
