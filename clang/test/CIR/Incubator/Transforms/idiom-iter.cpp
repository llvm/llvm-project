// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -I%S/../Inputs -fclangir-idiom-recognizer="remarks=found-calls" -clangir-verify-diagnostics %s -o %t.cir

namespace std {
template<typename T, unsigned N> struct array {
  T arr[N];
  struct iterator {
    T *p;
    constexpr explicit iterator(T *p) : p(p) {}
    constexpr bool operator!=(iterator o) { return p != o.p; }
    constexpr iterator &operator++() { ++p; return *this; }
    constexpr T &operator*() { return *p; }
  };
  constexpr iterator begin() { return iterator(arr); }
};
}

void iter_test()
{
  std::array<unsigned char, 3> v2 = {1, 2, 3};
  (void)v2.begin(); // no remark should be produced.
}