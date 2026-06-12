// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s -Wno-undefined-internal
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both                                             %s -Wno-undefined-internal

// both-no-diagnostics

namespace std {
inline namespace {
template <class _Tp> constexpr _Tp &&forward(_Tp &);
template <class _Tp> constexpr __remove_reference_t(_Tp) &&move(_Tp &&);
template <decltype(sizeof(int)), class> struct tuple_element;
template <class> struct tuple_size;
template <class, class _T2> struct pair {
  _T2 second;
};
template <class _T1, class _T2> pair<_T1, _T2> make_pair(_T1, _T2);
template <class _T1, class _T2> struct tuple_size<pair<_T1, _T2>> {
  static const int value = 2;
};
template <class _T1, class _T2> struct tuple_element<0, pair<_T1, _T2>> {
  using type = _T1;
};
template <class _T1, class _T2> struct tuple_element<1, pair<_T1, _T2>> {
  using type = _T2;
};
template <int> struct __get_pair {
  template <class _T1, class _T2> static _T1 &&get(pair<_T1, _T2>);
};
template <> struct __get_pair<1> {
  template <class _T1, class _T2>
  static constexpr _T2 &&get(pair<_T1, _T2> &&__p) {
    return std::forward(__p.second);
  }
};
template <int _Ip, class _T1, class _T2>
constexpr tuple_element<_Ip, pair<_T1, _T2>>::type &&get(pair<_T1, _T2> &&__p) {
  return __get_pair<_Ip>::get(std::move(__p));
}
namespace _Algorithm {
struct __for_each;
}
template <class> struct __single_range;
template <class...> struct __specialized_algorithm;
template <class, class> using for_each_result = int;
struct {
  template <typename _Range, class _Func>
  for_each_result<_Range, _Func> operator()(_Range __range, _Func __func) {
    using _SpecialAlg =
        __specialized_algorithm<_Algorithm::__for_each, __single_range<_Range>>;
    _SpecialAlg()(__range, __func, 0);
    return {};
  }
} for_each;
template <class, class, class> struct __tree;
template <class _Tp, class _Compare, class _Allocator>
struct __specialized_algorithm<
    _Algorithm::__for_each, __single_range<__tree<_Tp, _Compare, _Allocator>>> {
  template <class _Tree, class _Func, class _Proj>
  auto operator()(_Tree, _Func __func, _Proj) {
    return make_pair(0, __func);
  }
};
template <class, class, class = int, class = int> struct map {
  typedef __tree<int, int, int> __base;
};
template <class _Key, class _Tp, class _Compare, class _Allocator>
struct __specialized_algorithm<
    _Algorithm::__for_each,
    __single_range<map<_Key, _Tp, _Compare, _Allocator>>> {
  template <class _Map, class _Func, class _Proj>
  auto operator()(_Map __map, _Func __func, _Proj) {
    auto [_, __func2] = __specialized_algorithm<
        _Algorithm::__for_each,
        __single_range<typename map<_Compare, _Allocator>::__base>>()(
        __map, __func, 0);
    return __func2;
  }
};
} // namespace
} // namespace std


template <class Converter> void test_node_container(Converter) {
  std::map<int, int> c;
  int invoke_count;
  std::for_each(c, [&invoke_count] {});
}
void test() {
  test_node_container([] {});
}
