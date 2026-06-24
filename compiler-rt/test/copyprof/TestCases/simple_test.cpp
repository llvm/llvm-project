// RUN: %clangxx_copyprof -O0 -fcopyprof-static-size-threshold=24 %s -o %t
// RUN: %env_copyprof_opts=must_allocate=0:obj_size_threshold=16 %run %t 2>&1 | FileCheck %s --dump-input=always

#include <cstdint>
#include <memory>
#include <string.h>

void DoNotOptimize(void *ptr) {
  __asm__ __volatile__("" : : "r"(ptr) : "memory");
}

// Non-trivial class that is smaller than the threshold, i.e. should never
// generate reports.
struct NonTrivialTooSmall {
  NonTrivialTooSmall() {}
  ~NonTrivialTooSmall() {}
  NonTrivialTooSmall(const NonTrivialTooSmall &other)
      : x(other.x), y(other.y) {}
  NonTrivialTooSmall &operator=(const NonTrivialTooSmall &other) {
    x = other.x;
    y = other.y;
    return *this;
  }

  int x, y;
};

void NoReportTooSmall() {
  NonTrivialTooSmall t1;
  NonTrivialTooSmall t2 = t1;
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy

  NonTrivialTooSmall t3;
  t3 = t1;
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy
}

// Non-trivial class that is larger than the threshold.
struct NonTrivialWombat {
  NonTrivialWombat() {}
  ~NonTrivialWombat() {}
  NonTrivialWombat(const NonTrivialWombat &other)
      : x(other.x), y(other.y), z(other.z) {}
  NonTrivialWombat &operator=(const NonTrivialWombat &other) {
    x = other.x;
    y = other.y;
    z = other.z;
    return *this;
  }

  uint64_t x = 1;
  uint64_t y = 1;
  uint64_t z = 1;
};

void ReportUnnecessaryCopy() {
  NonTrivialWombat w;
  NonTrivialWombat w2 = w;
  // CHECK: [copyprof] Destroyed unnecessary copy amounting to 24 bytes
}

void NoReportWhenModified() {
  NonTrivialWombat w;
  NonTrivialWombat w2 = w;

  w2.x = 777;
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy
}

void NoReportMemset() {
  NonTrivialWombat w;
  NonTrivialWombat w2 = w;

  // TODO: b/464197306 - make the size invisible to the compiler so it cannot
  // replace the intrinsic w/ individual stores. Remove once fixed.
  int size = sizeof(w2);
  DoNotOptimize(&size);
  memset(reinterpret_cast<void *>(&w2), 0x66, size);
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy
}

void NoReportMemcpy() {
  NonTrivialWombat w;
  NonTrivialWombat w2 = w;

  // TODO: b/464197306 - make the size invisible to the compiler so it cannot
  // replace the intrinsic w/ individual stores. Remove once fixed.
  int size = sizeof(w2);
  DoNotOptimize(&size);
  int source = 0x666;
  memcpy(reinterpret_cast<void *>(&w2), &source, size);
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy
}

void NoReportMemmove() {
  NonTrivialWombat w;
  NonTrivialWombat w2 = w;

  // TODO: b/464197306 - make the size invisible to the compiler so it cannot
  // replace the intrinsic w/ individual stores. Remove once fixed.
  int size = sizeof(w2);
  DoNotOptimize(&size);
  int source = 0x666;
  memmove(reinterpret_cast<void *>(&w2), &source, size);
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy
}

int main(int argc, char **argv) {
  NoReportTooSmall();
  ReportUnnecessaryCopy();
  NoReportWhenModified();
  NoReportMemset();
  NoReportMemcpy();
  NoReportMemmove();

  return 0;
}
