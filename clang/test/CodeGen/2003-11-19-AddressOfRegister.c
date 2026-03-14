// RUN: %clang_cc1 %s -emit-llvm -o /dev/null

struct item {
  short delta[4];
};

int TEST(int nt) {
 register struct item *aa;
 aa[nt].delta;
 return 1;
}
