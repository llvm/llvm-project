// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both %s

// both-no-diagnostics

namespace std {
inline namespace __1 {
template <class _Tp> class unique_ptr;
template <class _Tp> class unique_ptr<_Tp[]> {
public:
  _Tp* __ptr_;

public:
  constexpr _Tp&
  operator[](unsigned i) const {
    return __ptr_[i];
  };
};
} // namespace __1
} // namespace std
struct WithTrivialDtor {
  int x = 6;
  constexpr friend void operator==(WithTrivialDtor const &x,
                                   WithTrivialDtor const &y) {
    (void)(x.x == y.x);
  }
};
constexpr bool test() {

  WithTrivialDtor array[50];
  std::unique_ptr<WithTrivialDtor[]> p(&array[0]);
  (void)(p[1] == WithTrivialDtor());

  return true;
}
static_assert(test());
