// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -std=c++20 %s -verify -fmodules -fmodules-cache-path=%t
// expected-no-diagnostics

#pragma clang module build std
module std {
  module concepts {}
  module functional {}
}
#pragma clang module contents
#pragma clang module begin std

template <class _Tp> struct common_reference {
  using type = _Tp;
};

#pragma clang module end
#pragma clang module begin std.concepts
#pragma clang module import std

template <class _Tp>
concept same_as = __is_same(_Tp, _Tp);

template <class _Tp>
concept common_reference_with =
    same_as<typename common_reference<_Tp>::type>;

#pragma clang module end
#pragma clang module begin std.functional
#pragma clang module import std.concepts

template <class, class _Ip>
concept sentinel_for = common_reference_with<_Ip>;

constexpr bool ntsf_subsumes_sf(sentinel_for<char *> auto)
  requires true
{
  return true;
}
bool ntsf_subsumes_sf(sentinel_for<char *> auto);
static_assert(ntsf_subsumes_sf(""));

#pragma clang module end
#pragma clang module endbuild
