// TODO(jannewger): replace with a proper profile file test using pre-created
// profile files passed to llvm-profdata (similar to what memprof does).
//
// RUN: %clangxx_copyprof -O0 %s -o %t
// RUN: %env_copyprof_opts=obj_size_threshold=47:report_output_file=%t.report %run %t 2>&1 | FileCheck %s --check-prefix=STDOUT --allow-empty
// RUN: rm %t.report.*

#include <cstddef>
#include <cstring>
#include <memory>

namespace {

// Simple class to triggers CopyProf reports.
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

void TriggerReport() {
  WombatAlloc large_alloc(48);
  WombatAlloc copy = large_alloc;
  // Verify that nothing is printed to stdout when report_output_file is set.
  // STDOUT-NOT: [copyprof]
}

} // namespace

int main() {
  TriggerReport();
  return 0;
}
