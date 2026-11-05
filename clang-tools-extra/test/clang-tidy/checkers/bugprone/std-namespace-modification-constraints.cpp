// RUN: %check_clang_tidy -std=c++20-or-later %s bugprone-std-namespace-modification %t \
// RUN:   -- -- -isystem %S/../Inputs/Headers

#include "system-header-simulation.h"

namespace prog {
template <class...> struct Foo {};

template <class> struct IsFoo : std::false_type {};
template <class... Args> struct IsFoo<Foo<Args...>> : std::true_type {};

struct Base {};

template <class T> concept FooType = IsFoo<T>::value;
template <class T> concept Anything = true;
template <class T> concept Big = sizeof(T) > 4;
template <class T> concept NotFoo = !FooType<T>;
template <class T> concept FooAndBig = FooType<T> && Big<T>;
// Mentions a program-defined type, but does not constrain the parameter.
template <class T> concept UnrelatedFoo = IsFoo<int>::value;
template <class T, class U> concept FooSecond = IsFoo<U>::value;

template <class T> inline constexpr bool IsFooV = IsFoo<T>::value;
template <class T> constexpr bool isFoo() { return IsFoo<T>::value; }
template <class T> concept FooTypeV = IsFooV<T>;

template <class> inline constexpr bool AlwaysV = true;
template <class T> inline constexpr bool ChainAlwaysV = AlwaysV<T>;
template <class T> inline constexpr bool ChainFooV = IsFooV<T>;
template <class T> inline constexpr bool ConceptV = FooType<T>;
template <class T, class U> inline constexpr bool SecondV = IsFoo<U>::value;
template <int N, class T> inline constexpr bool NonTypeFirstV = IsFoo<T>::value;

template <class> inline constexpr bool PartialSpecV = false;
template <class... Args>
inline constexpr bool PartialSpecV<Foo<Args...>> = true;
template <class> inline constexpr bool ExplicitSpecV = false;
template <> inline constexpr bool ExplicitSpecV<Foo<int>> = true;
template <class T>
inline constexpr bool RecursiveV = IsFoo<T>::value || RecursiveV<T *>;
template <class... Ts> inline constexpr bool PackV = (IsFoo<Ts>::value && ...);

// Declared before it is defined.
template <class T> extern const bool LaterAlwaysV;
template <class T> const bool LaterAlwaysV = true;

template <template <class> class TT, class T> concept Applies = TT<T>::value;
} // namespace prog

// The constraints are evaluated for probe types that are not program-defined,
// such as int, double, void, int * and std::tuple<int>. If no probe satisfies
// them, the specialization depends on a program-defined type.

// no-warning: only prog::Foo<...> satisfies the constraint.
template <prog::FooType T>
struct std::hash<T> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <prog::Anything T>
struct std::hash<T *> {};

// no-warning
template <class T> requires prog::FooType<T> && prog::Big<T>
struct std::hash<T &> {};

// no-warning: nested conjunction through a concept definition.
template <class T> requires prog::FooAndBig<T>
struct std::hash<T &&> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires prog::Anything<T> && prog::Big<T>
struct std::hash<const T> {};

// no-warning
template <class T> requires prog::FooType<T> || prog::FooAndBig<T>
struct std::hash<volatile T> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires prog::FooType<T> || prog::Big<T>
struct std::hash<T[1]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires std::true_type::value || prog::FooType<T>
struct std::hash<T[2]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <prog::NotFoo T>
struct std::hash<T[3]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires (!prog::FooType<T>)
struct std::hash<T[4]> {};

// no-warning
template <std::derived_from<prog::Base> T>
struct std::hash<T[5]> {};

// Constraints that no type satisfies cannot admit a non-program-defined type.
// no-warning
template <std::derived_from<int> T>
struct std::hash<T[6]> {};

// no-warning
template <class T> requires prog::IsFoo<T>::value
struct std::hash<T[7]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires std::true_type::value
struct std::hash<T[8]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T>
struct std::hash<T[9]> {};

// no-warning: not satisfiable.
template <class T> requires prog::IsFoo<int>::value
struct std::hash<T[10]> {};

// no-warning: not satisfiable.
template <prog::UnrelatedFoo T>
struct std::hash<T[11]> {};

// no-warning: not satisfiable.
template <class T> requires std::derived_from<int, prog::Base>
struct std::hash<T[12]> {};

// no-warning: not satisfiable.
template <class T> requires prog::FooSecond<T, int>
struct std::hash<T[13]> {};

// no-warning
template <class T> requires prog::FooSecond<int, T>
struct std::hash<T[14]> {};

// no-warning
template <class T> requires prog::FooType<T> && prog::IsFoo<int>::value
struct std::hash<T[15]> {};

// Non-type template parameters are probed with 0 and 1.
// no-warning
template <__SIZE_TYPE__ I, prog::FooType T>
struct std::tuple_element<I, T> {
  using type = int;
};

// no-warning
template <__SIZE_TYPE__ I, class T> requires prog::FooType<T>
struct std::tuple_element<I, const T> {
  using type = int;
};

// A constraint on the non-type parameter only does not restrict the type.
// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <__SIZE_TYPE__ I, class T> requires (I < 2)
struct std::tuple_element<I, volatile T> {
  using type = int;
};

// no-warning
template <prog::FooType T>
struct std::tuple_size<T> : std::integral_constant<__SIZE_TYPE__, 2> {};

// no-warning
template <class T> requires prog::IsFooV<T>
struct std::hash<T[16]> {};

// no-warning
template <class T> requires (prog::isFoo<T>())
struct std::hash<T[17]> {};

// no-warning: through a program-defined concept.
template <prog::FooTypeV T>
struct std::hash<T[18]> {};

// no-warning: not satisfiable.
template <class T> requires prog::IsFooV<int>
struct std::hash<T[19]> {};

// no-warning: not satisfiable.
template <class T> requires (prog::isFoo<int>())
struct std::hash<T[20]> {};

// FIXME: std::io_errc satisfies this, but it is not a probe type. Enumeration
// probes are only used if std::byte is declared.
// no-warning
template <class T> requires std::is_enum_v<T>
struct std::hash<T[21]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires prog::AlwaysV<T>
struct std::hash<T[22]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires prog::ChainAlwaysV<T>
struct std::hash<T[23]> {};

// no-warning
template <class T> requires prog::ChainFooV<T>
struct std::hash<T[24]> {};

// no-warning
template <class T> requires prog::ConceptV<T>
struct std::hash<T[25]> {};

// no-warning: not satisfiable.
template <class T> requires prog::SecondV<T, int>
struct std::hash<T[26]> {};

// no-warning
template <class T> requires prog::SecondV<int, T>
struct std::hash<T[27]> {};

// no-warning
template <class T> requires prog::NonTypeFirstV<1, T>
struct std::hash<T[28]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires (!prog::ChainFooV<T>)
struct std::hash<T[29]> {};

// no-warning
template <class T> requires prog::PartialSpecV<T>
struct std::hash<T[30]> {};

// no-warning
template <class T> requires prog::ExplicitSpecV<T>
struct std::hash<T[31]> {};

// no-warning: instantiating RecursiveV fails for every probe.
template <class T> requires prog::RecursiveV<T>
struct std::hash<T[32]> {};

// no-warning
template <class T> requires prog::PackV<T>
struct std::hash<T[33]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires prog::LaterAlwaysV<T>
struct std::hash<T[34]> {};

// FIXME: std::is_error_code_enum is only true for std::io_errc and short, and
// std::io_errc is not a probe type.
// no-warning
template <class T> requires prog::Applies<std::is_error_code_enum, T>
struct std::hash<T[35]> {};

// no-warning
template <class T> requires prog::Applies<prog::IsFoo, T>
struct std::hash<T[36]> {};

// IsFoo<int> has a member 'type', inherited from std::false_type.
// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires requires { typename prog::IsFoo<T>::type; }
struct std::hash<T[37]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires requires { typename T::type; }
struct std::hash<T[38]> {};

// no-warning
template <class T> requires prog::FooType<T> && (!prog::Big<T>)
struct std::hash<T[39]> {};

// Packs are probed with an empty pack and with one probe type. A fold over '&&'
// is satisfied by an empty pack, e.g. std::tuple<>.
// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class... Ts> requires (prog::FooType<Ts> && ...)
struct std::hash<std::tuple<Ts...>> {};

// no-warning
template <class... Ts> requires (prog::FooType<Ts> || ...)
struct std::hash<volatile std::tuple<Ts...>> {};

// no-warning
template <class T, class... Ts>
  requires (prog::FooType<T> && ... && prog::FooType<Ts>)
struct std::hash<std::tuple<T, Ts...> *> {};

// CHECK-MESSAGES: :[[@LINE+3]]:13: warning: modification of 'std' namespace
template <class T, class... Ts>
  requires (prog::FooType<Ts> || ... || prog::Big<T>)
struct std::hash<std::tuple<T, Ts...> &> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class... Ts> requires (std::is_enum_v<Ts> && ...)
struct std::hash<const std::tuple<Ts...>> {};

// A constrained parameter pack is a fold over '&&' and is satisfied by an
// empty pack as well.
// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <prog::FooType... Ts>
struct std::hash<const volatile std::tuple<Ts...>> {};

// no-warning
template <prog::FooType T, prog::FooType... Ts>
struct std::hash<const std::tuple<T, Ts...> *> {};

// sizeof(bool) is 1, like sizeof(char).
// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires (sizeof(T) == sizeof(prog::IsFooV<int>))
struct std::hash<T[40]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires (sizeof(T) == sizeof(prog::isFoo<int>()))
struct std::hash<T[41]> {};

namespace prog {
template <class T> using FooAlias = IsFoo<T>;
struct Callback {
  void operator()(Foo<>) const;
};
template <class T> struct Box {
  T Member;
};
} // namespace prog

// no-warning: an alias template of a program-defined trait.
template <class T> requires prog::FooAlias<T>::value
struct std::hash<T[42]> {};

// no-warning: the callback only accepts a program-defined type.
template <class T> requires requires(prog::Callback C, T V) { C(V); }
struct std::hash<T[43]> {};

// sizeof(prog::Base) is 1, like sizeof(char).
// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires (sizeof(T) == sizeof(prog::Base))
struct std::hash<T[44]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires (prog::IsFoo<T>::value == false)
struct std::hash<T[45]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires requires { requires !prog::FooType<T>; }
struct std::hash<T[46]> {};

// no-warning
template <class T> requires requires { requires prog::FooType<T>; }
struct std::hash<T[47]> {};

// Errors while instantiating templates for a probe type (here Box<void>) are
// not reported, and the probe does not satisfy the constraints.
// no-warning
template <class T> requires (sizeof(prog::Box<T>) > 0) && prog::FooType<T>
struct std::hash<T[48]> {};

// CHECK-MESSAGES: :[[@LINE+2]]:13: warning: modification of 'std' namespace
template <class T> requires (sizeof(prog::Box<T>) > 0)
struct std::hash<T[49]> {};
