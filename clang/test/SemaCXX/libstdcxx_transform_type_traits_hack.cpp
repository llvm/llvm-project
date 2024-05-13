// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++11 -fms-extensions -Wno-microsoft %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++14 -fms-extensions -Wno-microsoft %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -std=gnu++1z -fms-extensions -Wno-microsoft %s

// libstdc++ uses __remove_cv as an alias, so we make our transform type traits alias-revertible
template <class T, class U>
struct Same {
  static constexpr auto value = __is_same(T, U);
};

template <class T>
using __remove_const = int; // expected-warning{{keyword '__remove_const' will be made available as an identifier here}}
template <class T>
using A = Same<__remove_const<T>, __remove_const<T>>;

template <class T>
using __remove_restrict = int; // expected-warning{{keyword '__remove_restrict' will be made available as an identifier here}}
template <class T>
using B = Same<__remove_restrict<T>, __remove_restrict<T>>;

template <class T>
using __remove_volatile = int; // expected-warning{{keyword '__remove_volatile' will be made available as an identifier here}}
template <class T>
using C = Same<__remove_volatile<T>, __remove_volatile<T>>;

template <class T>
using __remove_cv = int; // expected-warning{{keyword '__remove_cv' will be made available as an identifier here}}
template <class T>
using D = Same<__remove_cv<T>, __remove_cv<T>>;

template <class T>
using __add_pointer = int; // expected-warning{{keyword '__add_pointer' will be made available as an identifier here}}
template <class T>
using E = Same<__add_pointer<T>, __add_pointer<T>>;

template <class T>
using __remove_pointer = int; // expected-warning{{keyword '__remove_pointer' will be made available as an identifier here}}
template <class T>
using F = Same<__remove_pointer<T>, __remove_pointer<T>>;

template <class T>
using __add_lvalue_reference = int; // expected-warning{{keyword '__add_lvalue_reference' will be made available as an identifier here}}
template <class T>
using G = Same<__add_lvalue_reference<T>, __add_lvalue_reference<T>>;

template <class T>
using __add_rvalue_reference = int; // expected-warning{{keyword '__add_rvalue_reference' will be made available as an identifier here}}
template <class T>
using H = Same<__add_rvalue_reference<T>, __add_rvalue_reference<T>>;

template <class T>
using __remove_reference_t = int; // expected-warning{{keyword '__remove_reference_t' will be made available as an identifier here}}
template <class T>
using I = Same<__remove_reference_t<T>, __remove_reference_t<T>>;

template <class T>
using __remove_cvref = int; // expected-warning{{keyword '__remove_cvref' will be made available as an identifier here}}
template <class T>
using J = Same<__remove_cvref<T>, __remove_cvref<T>>;

template <class T>
using __decay = int; // expected-warning{{keyword '__decay' will be made available as an identifier here}}
template <class T>
using K = Same<__decay<T>, __decay<T>>;

template <class T>
using __make_signed = int; // expected-warning{{keyword '__make_signed' will be made available as an identifier here}}
template <class T>
using L = Same<__make_signed<T>, __make_signed<T>>;

template <class T>
using __make_unsigned = int; // expected-warning{{keyword '__make_unsigned' will be made available as an identifier here}}
template <class T>
using M = Same<__make_unsigned<T>, __make_unsigned<T>>;

template <class T>
using __remove_extent = int; // expected-warning{{keyword '__remove_extent' will be made available as an identifier here}}
template <class T>
using N = Same<__remove_extent<T>, __remove_extent<T>>;

template <class T>
using __remove_all_extents = int; // expected-warning{{keyword '__remove_all_extents' will be made available as an identifier here}}
template <class T>
using O = Same<__remove_all_extents<T>, __remove_all_extents<T>>;
