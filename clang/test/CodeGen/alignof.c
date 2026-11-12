// RUN: %clang_cc1 -triple i386-unknown-unknown -O1 -emit-llvm -o %t %s
// RUN: grep 'ret i32 4' %t

enum e0 { E0 };
struct s0 {
  enum e0         a:31;
};

struct s0 t1_tmp;
int f0(void) {
  return __alignof__(t1_tmp);
}
