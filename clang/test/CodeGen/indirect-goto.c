// RUN: %clang_cc1 -triple i386-unknown-unknown -O3 -emit-llvm -o - %s | grep "ret i32 2520"

static int foo(unsigned i) {
  void *addrs[] = { &&L1, &&L2, &&L3, &&L4, &&L5 };
  int res = 1;

  goto *addrs[i];
 L5: res *= 11;
 L4: res *= 7;
 L3: res *= 5;
 L2: res *= 3;
 L1: res *= 2; 
  return res;
}

static int foo2(unsigned i) {
  static const void *addrs[] = { &&L1, &&L2, &&L3, &&L4, &&L5 };
  int res = 1;
  
  goto *addrs[i];
L5: res *= 11;
L4: res *= 7;
L3: res *= 5;
L2: res *= 3;
L1: res *= 2; 
  return res;
}

int main() {
  return foo(3)+foo2(4);
}
