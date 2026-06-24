// RUN: %clangxx_copyprof -O0 %s -o %t
// RUN: %env_copyprof_opts=must_allocate=1:obj_size_threshold=47 %run %t 2>&1 | FileCheck %s --dump-input=always
// RUN: %clangxx_copyprof -O1 %s -o %t
// RUN: %env_copyprof_opts=must_allocate=1:obj_size_threshold=47 %run %t 2>&1 | FileCheck %s --dump-input=always
// RUN: %clangxx_copyprof -O2 %s -o %t
// RUN: %env_copyprof_opts=must_allocate=1:obj_size_threshold=47 %run %t 2>&1 | FileCheck %s --dump-input=always

#include <cstddef>
#include <cstring>
#include <memory>

namespace {

struct WombatAlloc {
  // total_size is the total transitive size: sizeof(WombatAlloc) + allocated bytes
  explicit WombatAlloc(size_t total_size)
      : len_(total_size > sizeof(WombatAlloc) ? total_size - sizeof(WombatAlloc)
                                              : 0) {
    if (len_ > 0) {
      data_.reset(new char[len_]);
      std::memset(data_.get(), 0, len_);
    }
  }

  WombatAlloc(const WombatAlloc &other) : len_(other.len_) {
    if (len_ > 0) {
      data_.reset(new char[len_]);
      std::memcpy(data_.get(), other.data_.get(), len_);
    }
  }

  WombatAlloc &operator=(const WombatAlloc &other) {
    len_ = other.len_;
    if (len_ > 0) {
      data_.reset(new char[len_]);
      std::memcpy(data_.get(), other.data_.get(), len_);
    } else {
      data_.reset();
    }
    return *this;
  }

private:
  std::unique_ptr<char[]> data_;
  size_t len_;
};

// sizeof(WombatAlloc) = 16 bytes (8 for unique_ptr + 8 for size_t on 64-bit)
// Constructor argument = total transitive size (object + allocated memory)

void ReportUnnecessaryCopy() {
  // With threshold=47, objects > 47 bytes should be reported
  WombatAlloc small_alloc(32);
  WombatAlloc large_alloc(48);

  WombatAlloc small_copy = small_alloc;
  WombatAlloc large_copy = large_alloc;
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy
  // CHECK: [copyprof] Destroyed unnecessary copy amounting to 48 bytes
}

void ReportUnnecessaryCopyAssignment() {
  WombatAlloc small_alloc(32);
  WombatAlloc large_alloc(48);

  small_alloc = large_alloc;
  // CHECK: [copyprof] Destroyed unnecessary copy amounting to 48 bytes
}

void NoReportTooSmall() {
  WombatAlloc tiny_alloc(20);
  WombatAlloc small_alloc(32);

  WombatAlloc tiny_copy = tiny_alloc;
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy

  small_alloc = tiny_alloc;
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy
}

template <int TotalSize> void ReportCopyConstructor() {
  static_assert(TotalSize >= sizeof(WombatAlloc));
  WombatAlloc wombat(TotalSize);
  WombatAlloc copy = wombat;
}

void TestDifferentSizes() {
  ReportCopyConstructor<16>();
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy
  ReportCopyConstructor<17>();
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy
  ReportCopyConstructor<32>();
  // CHECK-NOT: [copyprof] Destroyed unnecessary copy
}

// Tests that precise tracking reports exact byte counts.
void TestPreciseTracking() {
  WombatAlloc precise_alloc(49);
  WombatAlloc copy = precise_alloc;
  // CHECK: [copyprof] Destroyed unnecessary copy amounting to 49 bytes
}

} // namespace

int main(int argc, char **argv) {
  ReportUnnecessaryCopy();
  ReportUnnecessaryCopyAssignment();
  NoReportTooSmall();
  TestDifferentSizes();
  TestPreciseTracking();

  return 0;
}
