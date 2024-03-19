// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-COUNT %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-COUNT-2 %s

void f1() {
  int a, b;
  register int c asm("r1");
  register int d asm("r2");

  // CHECK-COUNT: call i32 asm "lhi $0,5\0A", "={r1}"
  // CHECK-COUNT-2: call i32 asm "lhi $0,5\0A", "={r1}"
  __asm("lhi %0,5\n"
        : "={r1}"(a)
        :
        :);
  __asm("lhi %0,5\n"
        : "=r"(c)
        :
        :);

  // CHECK-COUNT: call i32 asm "lgr $0,$1\0A", "={r1},{r2}"
  // CHECK-COUNT-2: call i32 asm "lgr $0,$1\0A", "={r1},{r2}"
  __asm("lgr %0,%1\n"
        : "={r1}"(a)
        : "{r2}"(b)
        :);
  __asm("lgr %0,%1\n"
        : "=r"(c)
        : "r"(d)
        :);

  // CHECK-COUNT: call i32 asm "lgr $0,$1\0A", "={r1},{r2}"
  // CHECK-COUNT-2: call i32 asm "lgr $0,$1\0A", "={r1},{r2}"
  __asm("lgr %0,%1\n"
        : "={%r1}"(a)
        : "{%r2}"(b)
        :);
  __asm("lgr %0,%1\n"
        : "={r1}"(a)
        : "{%r2}"(b)
        :);

  // CHECK-COUNT: call i32 asm "lgr $0,$1\0A", "=&{r1},{r2}"
  // CHECK-COUNT-2: call i32 asm "lgr $0,$1\0A", "=&{r1},{r2}"
  __asm("lgr %0,%1\n"
        : "=&{r1}"(a)
        : "{%r2}"(b)
        :);
  __asm("lgr %0,%1\n"
        : "=&r"(c)
        : "r"(d)
        :);
}
