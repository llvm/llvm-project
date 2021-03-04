//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// type_traits
// common_reference

#include <type_traits>

using std::common_reference;
using std::common_reference_t;
using std::is_same_v;
using std::void_t;

template <class T>
constexpr bool has_type = requires {
  typename T::type;
};

// A slightly simplified variation of std::tuple
template <class...>
struct Tuple {};

template <class, class, class>
struct Tuple_helper {};
template <class... Ts, class... Us>
struct Tuple_helper<void_t<common_reference_t<Ts, Us>...>, Tuple<Ts...>,
                    Tuple<Us...> > {
  using type = Tuple<common_reference_t<Ts, Us>...>;
};

namespace std {
template <class... Ts, class... Us, template <class> class TQual,
          template <class> class UQual>
struct basic_common_reference< ::Tuple<Ts...>, ::Tuple<Us...>, TQual, UQual>
    : ::Tuple_helper<void, Tuple<TQual<Ts>...>, Tuple<UQual<Us>...> > {};
} // namespace std

struct X2 {};
struct Y2 {};
struct Z2 {};

namespace std {
template <>
struct common_type<X2, Y2> {
  using type = Z2;
};
template <>
struct common_type<Y2, X2> {
  using type = Z2;
};
} // namespace std

// (6.1)
//  -- If sizeof...(T) is zero, there shall be no member type.
static_assert(!has_type<common_reference<> >);

// (6.2)
//  -- Otherwise, if sizeof...(T) is one, let T0 denote the sole type in the
//     pack T. The member typedef type shall denote the same type as T0.
static_assert(is_same_v<common_reference_t<void>, void>);
static_assert(is_same_v<common_reference_t<int>, int>);
static_assert(is_same_v<common_reference_t<int&>, int&>);
static_assert(is_same_v<common_reference_t<int&&>, int&&>);
static_assert(is_same_v<common_reference_t<int const>, int const>);
static_assert(is_same_v<common_reference_t<int const&>, int const&>);
static_assert(is_same_v<common_reference_t<int const&&>, int const&&>);
static_assert(is_same_v<common_reference_t<int volatile[]>, int volatile[]>);
static_assert(
    is_same_v<common_reference_t<int volatile (&)[]>, int volatile (&)[]>);
static_assert(
    is_same_v<common_reference_t<int volatile(&&)[]>, int volatile(&&)[]>);
static_assert(is_same_v<common_reference_t<void (&)()>, void (&)()>);
static_assert(is_same_v<common_reference_t<void(&&)()>, void(&&)()>);

// (6.3)
//  -- Otherwise, if sizeof...(T) is two, let T1 and T2 denote the two types in
//     the pack T. Then
// (6.3.1)
//    -- If T1 and T2 are reference types and COMMON_REF(T1, T2) is well-formed,
//       then the member typedef type denotes that type.
struct B {};
struct D : B {};
static_assert(is_same_v<common_reference_t<B&, D&>, B&>);
static_assert(is_same_v<common_reference_t<B const&, D&>, B const&>);
static_assert(is_same_v<common_reference_t<B&, D const&>, B const&>);
static_assert(is_same_v<common_reference_t<B&, D const&, D&>, B const&>);
static_assert(is_same_v<common_reference_t<B&, D&, B&, D&>, B&>);

static_assert(is_same_v<common_reference_t<B&&, D&&>, B&&>);
static_assert(is_same_v<common_reference_t<B const&&, D&&>, B const&&>);
static_assert(is_same_v<common_reference_t<B&&, D const&&>, B const&&>);
static_assert(is_same_v<common_reference_t<B&, D&&>, B const&>);
static_assert(is_same_v<common_reference_t<B&, D const&&>, B const&>);
static_assert(is_same_v<common_reference_t<B const&, D&&>, B const&>);

static_assert(is_same_v<common_reference_t<B&&, D&>, B const&>);
static_assert(is_same_v<common_reference_t<B&&, D const&>, B const&>);
static_assert(is_same_v<common_reference_t<B const&&, D&>, B const&>);

static_assert(
    is_same_v<common_reference_t<int const volatile&&, int volatile&&>,
              int const volatile&&>);

static_assert(is_same_v<common_reference_t<int const&, int volatile&>,
                        int const volatile&>);

static_assert(
    is_same_v<common_reference_t<int (&)[10], int(&&)[10]>, int const (&)[10]>);

static_assert(
    is_same_v<common_reference_t<int const (&)[10], int volatile (&)[10]>,
              int const volatile (&)[10]>);

// (6.3.2)
//    -- Otherwise, if basic_common_reference<remove_cvref_t<T1>,
//       remove_cvref_t<T2>, XREF(T1), XREF(T2)>::type is well-formed, then the
//       member typedef type denotes that type.
static_assert(is_same_v<common_reference_t<const Tuple<int, short>&,
                                           Tuple<int&, short volatile&> >,
                        Tuple<const int&, const volatile short&> >);

static_assert(is_same_v<common_reference_t<volatile Tuple<int, short>&,
                                           const Tuple<int, short>&>,
                        const volatile Tuple<int, short>&>);

// (6.3.3)
//    -- Otherwise, if COND_RES(T1, T2) is well-formed, then the member typedef
//       type denotes that type.
static_assert(is_same_v<common_reference_t<void, void>, void>);
static_assert(is_same_v<common_reference_t<int, short>, int>);
static_assert(is_same_v<common_reference_t<int, short&>, int>);
static_assert(is_same_v<common_reference_t<int&, short&>, int>);
static_assert(is_same_v<common_reference_t<int&, short>, int>);

// tricky volatile reference case
static_assert(is_same_v<common_reference_t<int&&, int volatile&>, int>);
static_assert(is_same_v<common_reference_t<int volatile&, int&&>, int>);

static_assert(is_same_v<common_reference_t<int (&)[10], int (&)[11]>, int*>);

// https://github.com/ericniebler/stl2/issues/338
struct MyIntRef {
  MyIntRef(int&);
};
static_assert(is_same_v<common_reference_t<int&, MyIntRef>, MyIntRef>);

// (6.3.4)
//    -- Otherwise, if common_type_t<T1, T2> is well-formed, then the member
//       typedef type denotes that type.
struct moveonly {
  moveonly() = default;
  moveonly(moveonly&&) = default;
  moveonly& operator=(moveonly&&) = default;
};
struct moveonly2 : moveonly {};

static_assert(
    is_same_v<common_reference_t<moveonly const&, moveonly>, moveonly>);
static_assert(
    is_same_v<common_reference_t<moveonly2 const&, moveonly>, moveonly>);
static_assert(
    is_same_v<common_reference_t<moveonly const&, moveonly2>, moveonly>);

static_assert(is_same_v<common_reference_t<X2&, Y2 const&>, Z2>);

// (6.3.5)
//    -- Otherwise, there shall be no member type.
static_assert(!has_type<common_reference<volatile Tuple<short>&,
                                         const Tuple<int, short>&> >);

// (6.4)
//  -- Otherwise, if sizeof...(T) is greater than two, let T1, T2, and Rest,
//     respectively, denote the first, second, and (pack of) remaining types
//     comprising T. Let C be the type common_reference_t<T1, T2>. Then:
// (6.4.1)
//    -- If there is such a type C, the member typedef type shall denote the
//       same type, if any, as common_reference_t<C, Rest...>.
static_assert(is_same_v<common_reference_t<int, int, int>, int>);
static_assert(is_same_v<common_reference_t<int&&, int const&, int volatile&>,
                        int const volatile&>);
static_assert(is_same_v<common_reference_t<int&&, int const&, float&>, float>);

// (6.4.2)
//    -- Otherwise, there shall be no member type.
static_assert(!has_type<common_reference<int, short, int, char*> >);

int main(int, char**) { return 0; }
