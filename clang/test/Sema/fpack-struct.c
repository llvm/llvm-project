// RUN: %clang_cc1 -DEXPECTED_STRUCT_SIZE=5 -fpack-struct=1 %s
// RUN: %clang_cc1 -DEXPECTED_STRUCT_SIZE=6 -fpack-struct=2 %s

struct s0 {
       int x;
       char c;
};

int t0[sizeof(struct s0) == EXPECTED_STRUCT_SIZE ?: -1];
