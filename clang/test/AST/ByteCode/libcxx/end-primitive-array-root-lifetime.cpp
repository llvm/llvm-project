// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both %s

// both-no-diagnostics

namespace std {
inline namespace {
template <bool, class _IfRes, class> using conditional_t = _IfRes;
template <class _Ip>
concept input_iterator = requires { typename _Ip; };
auto end = int{};
namespace ranges {
template <class>
concept range = requires { end; };
template <class _Tp>
concept input_range = input_iterator<_Tp>;
template <class>
concept forward_range = false;
template <range _Rp> struct owning_view {
  _Rp __r_;
};
} // namespace ranges
template <int _Size> struct array {
  int __elems_[_Size];
};
template <class> struct allocator {
  constexpr array<2> *allocate(decltype(sizeof(int))) {
    return static_cast<array<2> *>(operator new(sizeof(array<2>)));
  }
};
namespace ranges {
template <input_range _View, forward_range _Pattern> struct join_with_view {
  join_with_view(_View, _Pattern);
};
} // namespace ranges
template <class> struct vector {
  constexpr ~vector() {
    (__end_ - 1)->~array<2>();
  }
  constexpr vector() {
    __end_ = __alloc_.allocate(0);
    _ConstructTransaction __tx(*this);
    ++__tx.__pos_;
  }
  array<2>* __end_;
  allocator<array<2>> __alloc_;
  struct _ConstructTransaction {
    constexpr _ConstructTransaction(vector &__v)
        : __v_(__v), __pos_(__v.__end_) {}
    constexpr ~_ConstructTransaction() { __v_.__end_ = __pos_; }
    vector __v_;
    array<2>* __pos_;
  };
};
} // namespace
} // namespace std
template <bool RefIsGlvalue, class Inner>
using VRange = std::conditional_t<RefIsGlvalue, std::vector<Inner>, Inner>;
template <int RefIsGlvalue> void test_pre_increment() {
  using V = VRange<RefIsGlvalue, std::array<2>>;
  using Pattern = std::array<2>;
  using JWV = std::ranges::join_with_view<std::ranges::owning_view<V>,
                                          std::ranges::owning_view<Pattern>>;
  JWV jwv({}, {});
}
