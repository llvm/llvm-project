//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <cassert>
#include <type_traits>
#include <vector>

struct CopyFromMutableReference {
  static int calls;
  int value;
  explicit CopyFromMutableReference(int v = 0) : value(v) {}
  CopyFromMutableReference(const CopyFromMutableReference&)            = default;
  CopyFromMutableReference& operator=(const CopyFromMutableReference&) = default;

  template <class U>
  CopyFromMutableReference(U& source) : value(source.value) {
    ++calls;
  }
};
int CopyFromMutableReference::calls = 0;

static_assert(std::is_trivially_copyable<CopyFromMutableReference>::value, "");
static_assert(std::is_trivially_copy_constructible<CopyFromMutableReference>::value, "");
static_assert(std::is_trivially_assignable<CopyFromMutableReference&, CopyFromMutableReference&>::value, "");
static_assert(!std::is_trivially_constructible<CopyFromMutableReference, CopyFromMutableReference&>::value, "");

int main(int, char**) {
  CopyFromMutableReference source[3];
  source[0].value                 = 1;
  source[1].value                 = 2;
  source[2].value                 = 3;
  CopyFromMutableReference::calls = 0;

  std::vector<CopyFromMutableReference> result(source, source + 3);
  assert(CopyFromMutableReference::calls == 3);
  assert(source[0].value == 1 && source[1].value == 2 && source[2].value == 3);
  assert(result.size() == 3);
  assert(result[0].value == 1 && result[1].value == 2 && result[2].value == 3);
  return 0;
}
