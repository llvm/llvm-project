// RUN: %clang_cc1 -std=c++2a -triple %itanium_abi_triple -emit-llvm -o - %s -w | FileCheck %s

template<class, int, class>
struct DummyType { };

inline void inline_func() {
  // CHECK: UlvE
  []{}();

  // CHECK: UlTyvE
  []<class>{}.operator()<int>();

  // CHECK: UlTyT_E
  []<class T>(T){}(1);

  // CHECK: UlTyTyT_T0_E
  []<class T1, class T2>(T1, T2){}(1, 2);

  // CHECK: UlTyTyT0_T_E
  []<class T1, class T2>(T2, T1){}(2, 1);

  // CHECK: UlTniTyTnjT0_E
  []<int I, class T, unsigned U>(T){}.operator()<1, int, 2>(3);

  // CHECK: UlTyTtTyTniTyETniTyvE
  []<class,
     template<class, int, class> class,
     int,
     class>{}.operator()<unsigned, DummyType, 5, int>();
}

void call_inline_func() {
  inline_func();
}

template<typename T, int> struct X {};

inline auto pack = []<typename ...T, T ...N>(T (&...)[N]) {};
int arr1[] = {1};
int arr2[] = {1, 2};
// CHECK: @_ZNK4packMUlTpTyTpTnT_DpRAT0__S_E_clIJiiEJLi1ELi2EEEEDaS2_(
void use_pack() { pack(arr1, arr2); }

inline void collision() {
  auto a = []<typename T, template<typename U, T> typename>{};
  auto b = []<typename T, template<typename U, U> typename>{};
  auto c = []<typename T, template<typename U, T> typename>{};
  a.operator()<int, X>();
  // CHECK: @_ZZ9collisionvENKUlTyTtTyTnT_EvE_clIi1XEEDav
  b.operator()<int, X>();
  // CHECK: @_ZZ9collisionvENKUlTyTtTyTnTL0__EvE_clIi1XEEDav
  c.operator()<int, X>();
  // CHECK: @_ZZ9collisionvENKUlTyTtTyTnT_EvE0_clIi1XEEDav
}
void use_collision() { collision(); }

namespace pack_not_pack_expansion {
  template<typename T, int, T...> struct X;
  // CHECK: @_ZNK23pack_not_pack_expansion1xMUlTyTtTyTnT_TpTnTL0__ETpTyvE_clIiNS_1XEJfEEEDav
  inline auto x = []<typename T, template<typename U, T, U...> typename, typename ...V>(){}; void f() { x.operator()<int, X, float>(); }
}

template<typename> void f() {
  // CHECK: define linkonce_odr {{.*}} @_ZZ1fIiEvvENKUlT_E_clIiEEDaS0_(
  auto x = [](auto){};
  x(0);
}
void use_f() { f<int>(); }

template<typename> struct Y {
  template<int> struct Z {};
};

template<typename ...T> void expanded() {
  auto x = []<T..., template<T> typename...>{};
  auto y = []<int, template<int> typename>{};
  auto z = []<int, int, template<int> typename, template<int> typename>{};
  // FIXME: Should we really require 'template' for y and z?
  x.template operator()<(T())..., Y<T>::template Z...>();
  y.template operator()<0, Y<int>::Z>();
  y.template operator()<1, Y<int>::Z>();
  z.template operator()<1, 2, Y<int>::Z, Y<float>::Z>();
}
void use_expanded() {
  // CHECK: @_ZZ8expandedIJEEvvENKUlvE_clIJEJEEEDav(
  // CHECK: @_ZZ8expandedIJEEvvENKUlTniTtTniEvE_clILi0EN1YIiE1ZEEEDav(
  // CHECK: @_ZZ8expandedIJEEvvENKUlTniTtTniEvE_clILi1EN1YIiE1ZEEEDav(
  // CHECK: @_ZZ8expandedIJEEvvENKUlTniTniTtTniETtTniEvE_clILi1ELi2EN1YIiE1ZENS2_IfE1ZEEEDav(
  expanded<>();

  // FIXME: Should we really be using J...E for arguments corresponding to an
  // expanded parameter pack?
  // Note that the <lambda-sig>s of 'x' and 'y' collide here, after pack expansion.
  // CHECK: @_ZZ8expandedIJiEEvvENKUlTniTtTniEvE_clIJLi0EEJN1YIiE1ZEEEEDav(
  // CHECK: @_ZZ8expandedIJiEEvvENKUlTniTtTniEvE0_clILi0EN1YIiE1ZEEEDav(
  // CHECK: @_ZZ8expandedIJiEEvvENKUlTniTtTniEvE0_clILi1EN1YIiE1ZEEEDav(
  // CHECK: @_ZZ8expandedIJiEEvvENKUlTniTniTtTniETtTniEvE_clILi1ELi2EN1YIiE1ZENS2_IfE1ZEEEDav(
  expanded<int>();

  // Note that the <lambda-sig>s of 'x' and 'z' collide here, after pack expansion.
  // CHECK: @_ZZ8expandedIJiiEEvvENKUlTniTniTtTniETtTniEvE_clIJLi0ELi0EEJN1YIiE1ZES4_EEEDav(
  // CHECK: @_ZZ8expandedIJiiEEvvENKUlTniTtTniEvE_clILi0EN1YIiE1ZEEEDav(
  // CHECK: @_ZZ8expandedIJiiEEvvENKUlTniTtTniEvE_clILi1EN1YIiE1ZEEEDav(
  // CHECK: @_ZZ8expandedIJiiEEvvENKUlTniTniTtTniETtTniEvE0_clILi1ELi2EN1YIiE1ZENS2_IfE1ZEEEDav(
  expanded<int, int>();
}
