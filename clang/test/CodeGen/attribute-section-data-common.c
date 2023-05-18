// RUN: %clang -S %s -o /dev/null
struct rtxc_snapshot {
  int a, b, c, d;
};
__attribute__ ((section("__DATA, __common"))) struct rtxc_snapshot rtxc_log_A[4];
