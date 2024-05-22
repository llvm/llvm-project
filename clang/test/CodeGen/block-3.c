// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 

int main(void) {
  ^{
                __attribute__((__blocks__(byref))) int index = ({ int __a; int __b; __a < __b ? __b : __a; });
   };
}

// PR13229
typedef struct {} Z;

typedef int (^B)(Z);

void testPR13229(void) {
  Z z1;
  B b1 = ^(Z z1) { return 1; };
  b1(z1);
}
