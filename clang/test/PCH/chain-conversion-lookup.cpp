// RUN: %clang_cc1 %s -emit-llvm -o - -chain-include %s -chain-include %s

#if !defined(PASS1)
#define PASS1
struct X {
  operator int*();
};

struct Z {
  operator int*();
};
#elif !defined(PASS2)
#define PASS2
struct Y {
  operator int *();
};
#else
int main() {
  X x;
  int *ip = x.operator int*();
  Y y;
  int *ip2 = y.operator int*();
  Z z;
  int *ip3 = z.operator int*();
}
#endif
