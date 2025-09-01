// RUN: %clangxx -fsanitize=alignment %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s --implicit-check-not="runtime error" -allow-empty
// Disable with msan and tsan because they also override global new
// UNSUPPORTED: ubsan-msan, ubsan-tsan

#include <cassert>
#include <cstddef>
#include <cstdlib>

void *operator new(std::size_t count) {
  constexpr const size_t offset = 8;

  // allocate a bit more so we can safely offset it
  void *ptr = std::malloc(count + offset);

  // verify malloc returned 16 bytes aligned mem
  static_assert(__STDCPP_DEFAULT_NEW_ALIGNMENT__ == 16,
                "Global new doesn't return 16 bytes aligned memory!");
  assert((reinterpret_cast<std::ptrdiff_t>(ptr) &
          (__STDCPP_DEFAULT_NEW_ALIGNMENT__ - 1)) == 0);

  return static_cast<char *>(ptr) + offset;
}

struct Param {
  void *_cookie1;
  void *_cookie2;
};

static_assert(alignof(Param) == 8, "Param struct alignment must be 8 bytes!");

int main() { Param *p = new Param; }
