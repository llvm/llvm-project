// RUN: %clangxx -std=c++14 %target_itanium_abi_host_triple -g %s -o %t.O0.out
// RUN: %test_debuginfo %s %t.O0.out

#include <memory>

static volatile int sink;

static void use_shared_ptr(std::shared_ptr<int> ptr) {
  // DEBUGGER: break 10
  sink = *ptr;
}

static void use_unique_ptr(std::unique_ptr<int> ptr) {
  // DEBUGGER: break 15
  sink = *ptr;
}

int main() {
  auto sp_1 = std::make_shared<int>(1234);
  use_shared_ptr(sp_1);

  auto up_1 = std::make_unique<int>(5678);
  use_unique_ptr(std::move(up_1));

  return 0;
}

// DEBUGGER: r

// (at line 10)
// DEBUGGER: p ptr
// CHECK: shared_ptr<int>
// CHECK-SAME: 1234

// DEBUGGER: c

// (at line 16)
// DEBUGGER: p ptr
// CHECK: unique_ptr<int>
// TODO: lldb's unique_ptr data formatter doesn't pretty-print its wrapped
// object.
