// RUN: %clang -O0 %s -target bpf -g -c -o /dev/null -fexperimental-new-pass-manager
// REQUIRES: bpf-registered-target

struct ss {
  int a;
};
int foo() { return __builtin_btf_type_id(0, 0) + __builtin_preserve_type_info(*(struct ss *)0, 0); }
