//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that vector diagnoses an allocator which has to implement rebind with an appropriate error message

#include <vector>

class FooAllocator {
public:
  using value_type = int;
  FooAllocator()   = default;

  int* allocate(int num_objects);

  void deallocate(int* ptr, int num_objects);

  bool operator==(const FooAllocator&) const { return true; }
  bool operator!=(const FooAllocator&) const { return false; }
};

void func() {
  std::vector<int, FooAllocator>
      v; //expected-error-re@*:* {{static assertion failed {{.*}}This allocator has to implement rebind}}
}
