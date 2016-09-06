// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: grep '@llvm.used = .*@a0' %t
// RUN: grep '@llvm.used = .*@g0' %t
// RUN: grep '@llvm.used = .*@f0' %t
// RUN: grep '@llvm.used = .*@f1.l0' %t


int g0 __attribute__((used));

static void __attribute__((used)) f0(void) {
}

void f1() { 
  static int l0 __attribute__((used)) = 5225; 
}

__attribute__((used)) int a0;
void pr27535() { (void)a0; }
