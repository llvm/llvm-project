// RUN: %clang -emit-llvm -S -o %t %s

struct s0 {
  void *a;
};
struct s0 * __attribute__((objc_gc(strong))) g0;
void f0(void) {
  g0->a = 0;
}
