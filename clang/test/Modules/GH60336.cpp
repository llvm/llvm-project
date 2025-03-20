// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -std=c++20 %s -verify -fmodules -fmodules-cache-path=%t
// expected-no-diagnostics

#pragma clang module build std
module std   [system] {
  module concepts     [system] {
      module assignable         [system] {
    }
    export *
  }
  module functional     [system] {
    export *
  }


  module type_traits     [system] {
    export *
  }
}

#pragma clang module contents
#pragma clang module begin std.type_traits
namespace std {
template<class _Tp, class _Up>
concept same_as = __is_same(_Tp, _Up);

template <class...>
struct common_reference;

template <class _Tp, class _Up> struct common_reference<_Tp, _Up>
{
    using type = _Tp;
};
}
#pragma clang module end // type_traits

#pragma clang module begin std.concepts.assignable
#pragma clang module import std.type_traits
namespace std {
template<class _Tp, class _Up>
concept common_reference_with =
  same_as<typename common_reference<_Tp, _Up>::type, typename common_reference<_Up, _Tp>::type>;
}
namespace std {
template<class _Lhs, class _Rhs>
concept assignable_from =
  common_reference_with<const __remove_reference_t(_Lhs)&, const __remove_reference_t(_Rhs)&> ;
}
#pragma clang module end // std.concepts.assignable

#pragma clang module begin std.functional
#pragma clang module import std.concepts.assignable
namespace std {
template<class _Sp, class _Ip>
concept sentinel_for = assignable_from<_Ip&, _Ip>;
template <class _Sp, class _Ip>
concept nothrow_sentinel_for = sentinel_for<_Sp, _Ip>;
}
#pragma clang module end   // std::functional
#pragma clang module endbuild // contents


#pragma clang module import std.functional
constexpr bool ntsf_subsumes_sf(std::nothrow_sentinel_for<char*> auto) requires true {
  return true;
}
constexpr bool ntsf_subsumes_sf(std::sentinel_for<char*> auto);
static_assert(ntsf_subsumes_sf("foo"));
