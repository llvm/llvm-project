// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both                                             %s

// both-no-diagnostics


template <class _Tp> struct __cw_fixed_value {
  constexpr __cw_fixed_value(_Tp __v) : __data(__v) {}
  _Tp __data;
};


template <__cw_fixed_value _Xp>
struct constant_wrapper {
  static constexpr auto &ref_value = _Xp.__data;
  template <typename _Rp>
  auto operator=(_Rp) -> constant_wrapper<(ref_value = _Rp::ref_value)>;
};

struct WithOps {
  int value;
  constexpr WithOps operator=(int i) const { return {i}; }
};

void test() {
  constexpr constant_wrapper<3> cw3;
  constant_wrapper<WithOps{}> cwOps5;
  auto result = cwOps5 = cw3;
  static_assert(result.ref_value.value);
}

