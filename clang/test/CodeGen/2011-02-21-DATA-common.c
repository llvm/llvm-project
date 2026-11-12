// RUN: %clang_cc1 -emit-llvm %s -o /dev/null
struct rtxc_snapshot {
  int a, b, c, d;
};
__attribute__ ((section("__DATA, __common"))) static struct rtxc_snapshot rtxc_log_A[4];
