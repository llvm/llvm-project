// RUN: %clang_cc1 -emit-llvm %s -o /dev/null
// This is a testcase for PR461
typedef struct {
  unsigned min_align: 1;
  unsigned : 1;
} addr_diff_vec_flags;

addr_diff_vec_flags X;
