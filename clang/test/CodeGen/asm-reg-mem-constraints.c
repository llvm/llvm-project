// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefixes=O0 %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -O2 %s -o - | FileCheck --check-prefixes=O2 %s

void test_1(unsigned long flags) {
  // O0-LABEL: @test_1
  // O0:         call void asm sideeffect "", "rm,~{dirflag},~{fpsr},~{flags}"(i32 %0)
  //
  // O2-LABEL: @test_1
  // O2:         call void asm sideeffect "", "rm,~{dirflag},~{fpsr},~{flags}"(i32 %flags)
  asm ("" : : "rm" (flags));
}

unsigned long test_2(void) {
  // O0-LABEL: @test_2
  // O0:         call void asm "", "=*rm,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %out)
  //
  // O2-LABEL: @test_2
  // O2:         %0 = tail call i32 asm "", "=rm,~{dirflag},~{fpsr},~{flags}"()
  unsigned long out;
  asm ("" : "=rm" (out));
  return out;
}

void test_3(unsigned long flags) {
  // O0-LABEL: @test_3
  // O0:         call void asm sideeffect "", "imr,~{dirflag},~{fpsr},~{flags}"(i32 %0)
  //
  // O2-LABEL: @test_3
  // O2:         call void asm sideeffect "", "imr,~{dirflag},~{fpsr},~{flags}"(i32 %flags)
  asm ("" : : "g" (flags));
}

unsigned long test_4(void) {
  // O0-LABEL: @test_4
  // O0:         call void asm "", "=*imr,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) %out)
  //
  // O2-LABEL: @test_4
  // O2:         %0 = tail call i32 asm "", "=imr,~{dirflag},~{fpsr},~{flags}"()
  unsigned long out;
  asm ("" : "=g" (out));
  return out;
}

void test_5(int len) {
  // O0-LABEL: @test_5
  // O0:         call void asm sideeffect "", "=*&rm,0,~{dirflag},~{fpsr},~{flags}"
  //
  // O2-LABEL: @test_5
  // O2:         %0 = tail call i32 asm sideeffect "", "=&rm,0,~{dirflag},~{fpsr},~{flags}"(i32 %len)
  __asm__ volatile ("" : "+&&rm" (len));
}

void test_6(int len) {
  // O0-LABEL: @test_6
  // O0:         call void asm sideeffect "", "=*%rm,=*rm,0,1,~{dirflag},~{fpsr},~{flags}"
  //
  // O2-LABEL: @test_6
  // O2:         %0 = tail call { i32, i32 } asm sideeffect "", "=%rm,=rm,0,1,~{dirflag},~{fpsr},~{flags}"(i32 %len, i32 %len)
  __asm__ volatile ("" : "+%%rm" (len), "+rm" (len));
}

// PR3908
void test_7(int r) {
  // O0-LABEL: @test_7
  // O0:         call i32 asm "# PR3908 $1 $3 $2 $0", "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags}"
  // O0-SAME:      (i32 0, i32 0, double 0.000000e+00, i32 %{{.*}})
  //
  // O2-LABEL: @test_7
  // O2:         %0 = tail call i32 asm "# PR3908 $1 $3 $2 $0", "=r,mx,mr,x,0,~{dirflag},~{fpsr},~{flags}"
  // O2-SAME:      (i32 0, i32 0, double 0.000000e+00, i32 %{{.*}})
  __asm__ ("# PR3908 %[lf] %[xx] %[li] %[r]"
           : [r] "+r" (r)
           : [lf] "mx" (0), [li] "mr" (0), [xx] "x" ((double)(0)));
}
