// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -o - %s | FileCheck --check-prefix=CHECK-COUNT %s

void f1() {
  int a, b;

  // CHECK-COUNT: call i32 asm "lhi $0,5\0A", "={r1}{r2}"
  __asm("lhi %0,5\n"
        : "={r1}{r2}"(a)
        :
        :);

  // CHECK-COUNT: call i32 asm "lgr $0,$1\0A", "={r1}{r3},{r2}{r4}"
  __asm("lgr %0,%1\n"
        : "={r1}{r3}"(a)
        : "{r2}{r4}"(b)
        :);

  // CHECK-COUNT: call i32 asm "lgr $0,$1\0A", "={r1}{r3},{r2}{r4}"
  __asm("lgr %0,%1\n"
        : "={%r1}{%r3}"(a)
        : "{%r2}{%r4}"(b)
        :);

  // CHECK-COUNT: call i32 asm "lgr $0,$1\0A", "=&{r1}{r3},{r2}"
  __asm("lgr %0,%1\n"
        : "=&{r1}{r3}"(a)
        : "{%r2}"(b)
        :);
  
  // CHECK-COUNT: call i32 asm "lgr $0,$1\0A", "=r{r1}{r3},{r2}r{r4}"
  __asm("lgr %0,%1\n"
        : "=r{r1}{r3}"(a)
        : "{r2}r{r4}"(b)
        :);
}
