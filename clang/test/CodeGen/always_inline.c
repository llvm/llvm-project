// RUN: %clang -emit-llvm -S -o %t %s
// RUN: not grep '@f0' %t
// RUN: not grep 'call ' %t
// RUN: %clang -Xclang -disable-llvm-passes -emit-llvm -S -o %t %s
// RUN: grep '@f0' %t | count 2

//static int f0() { 
static int __attribute__((always_inline)) f0() { 
  return 1;
}

int f1() {
  return f0();
}

// PR4372
inline int f2() __attribute__((always_inline));
int f2() { return 7; }
int f3(void) { return f2(); }

