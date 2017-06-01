// RUN: %clangxx_cfi -g -fsanitize-stats -o %t %s
// RUN: env SANITIZER_STATS_PATH=%t.stats %t
// RUN: sanstats %t.stats | FileCheck %s

// FIXME: We currently emit the wrong debug info under devirtualization.
// UNSUPPORTED: devirt

struct ABase {};

struct A : ABase {
  virtual void vf() {}
  void nvf() {}
};

extern "C" __attribute__((noinline)) void vcall(A *a) {
  // CHECK: stats.cpp:[[@LINE+1]] {{_?}}vcall cfi-vcall 37
  a->vf();
}

extern "C" __attribute__((noinline)) void nvcall(A *a) {
  // CHECK: stats.cpp:[[@LINE+1]] {{_?}}nvcall cfi-nvcall 51
  a->nvf();
}

extern "C" __attribute__((noinline)) A *dcast(A *a) {
  // CHECK: stats.cpp:[[@LINE+1]] {{_?}}dcast cfi-derived-cast 24
  return (A *)(ABase *)a;
}

extern "C" __attribute__((noinline)) A *ucast(A *a) {
  // CHECK: stats.cpp:[[@LINE+1]] {{_?}}ucast cfi-unrelated-cast 81
  return (A *)(char *)a;
}

extern "C" __attribute__((noinline)) void unreachable(A *a) {
  // CHECK-NOT: unreachable
  a->vf();
}

int main() {
  A a;
  for (unsigned i = 0; i != 37; ++i)
    vcall(&a);
  for (unsigned i = 0; i != 51; ++i)
    nvcall(&a);
  for (unsigned i = 0; i != 24; ++i)
    dcast(&a);
  for (unsigned i = 0; i != 81; ++i)
    ucast(&a);
  for (unsigned i = 0; i != 0; ++i)
    unreachable(&a);
}
