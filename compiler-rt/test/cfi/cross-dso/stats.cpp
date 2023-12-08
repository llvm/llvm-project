// RUN: %clangxx_cfi_dso -DSHARED_LIB -fPIC -g -fsanitize-stats -shared -o %t.so %s
// RUN: %clangxx_cfi_dso -g -fsanitize-stats -o %t %s %t.so
// RUN: env SANITIZER_STATS_PATH=%t.stats %t
// RUN: sanstats %t.stats | FileCheck %s

// CFI-icall is not implemented in thinlto mode => ".cfi" suffixes are missing
// in sanstats output.

// FIXME: %t.stats must be transferred from device to host for this to work on Android.
// XFAIL: android

struct ABase {};

struct A : ABase {
  virtual void vf() {}
  void nvf() {}
};

extern "C" void vcall(A *a);
extern "C" void nvcall(A *a);

#ifdef SHARED_LIB

extern "C" __attribute__((noinline)) void vcall(A *a) {
  // CHECK-DAG: stats.cpp:[[@LINE+1]] vcall.cfi cfi-vcall 37
  a->vf();
}

extern "C" __attribute__((noinline)) void nvcall(A *a) {
  // CHECK-DAG: stats.cpp:[[@LINE+1]] nvcall.cfi cfi-nvcall 51
  a->nvf();
}

#else

extern "C" __attribute__((noinline)) A *dcast(A *a) {
  // CHECK-DAG: stats.cpp:[[@LINE+1]] dcast.cfi cfi-derived-cast 24
  return (A *)(ABase *)a;
}

extern "C" __attribute__((noinline)) A *ucast(A *a) {
  // CHECK-DAG: stats.cpp:[[@LINE+1]] ucast.cfi cfi-unrelated-cast 81
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

#endif
