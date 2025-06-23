// RUN: %check_clang_tidy -std=c++20 %s bugprone-exception-escape %t -- \
// RUN:     -- -fexceptions -Wno-everything

namespace GH104457 {

consteval int consteval_fn(int a) {
  if (a == 0)
    throw 1;
  return a;
}

int test() noexcept { return consteval_fn(1); }

} // namespace GH104457
