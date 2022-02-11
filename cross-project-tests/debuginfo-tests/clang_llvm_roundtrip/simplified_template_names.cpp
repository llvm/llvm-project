// RUN: %clang %target_itanium_abi_host_triple %s -c -o - -gdwarf-4 -Xclang -gsimple-template-names=mangled -Xclang -debug-forward-template-params -std=c++20 \
// RUN:   | llvm-dwarfdump --verify -
// RUN: %clang %target_itanium_abi_host_triple %s -c -o - -gdwarf-4 -Xclang -gsimple-template-names=mangled -Xclang -debug-forward-template-params -std=c++20 -fdebug-types-section \
// RUN:   | llvm-dwarfdump --verify -
// RUN: %clang %target_itanium_abi_host_triple %s -c -o - -gdwarf-5 -Xclang -gsimple-template-names=mangled -Xclang -debug-forward-template-params -std=c++20 -fdebug-types-section \
// RUN:   | llvm-dwarfdump --verify -
#include <cstdint>
template<typename ...Ts>
struct t1 {
};
template<typename ...Ts>
struct t2;
struct udt {
};
namespace ns {
struct udt {
};
namespace inner {
template<typename T> struct ttp { };
struct udt { };
}
template<template<typename> class T>
void ttp_user() { }
enum Enumeration { Enumerator1, Enumerator2, Enumerator3 = 1 };
enum class EnumerationClass { Enumerator1, Enumerator2, Enumerator3 = 1 };
enum { AnonEnum1, AnonEnum2, AnonEnum3 = 1 };
enum EnumerationSmall : unsigned char { kNeg = 0xff };
}
template <typename... Ts>
void f1() {
  t1<Ts...> v1;
  t2<Ts...> *v2;
}
template<bool b, int i>
void f2() {
}
template<typename T, T ...A>
void f3() {
}
template<typename T, unsigned = 3>
void f4() {
}
template<typename T, bool b = false>
struct t3 { };
extern template class t3<int>;
template class t3<int>;
struct outer_class {
  struct inner_class {
  };
};
int i = 3;
template<unsigned N>
struct t4 { };
namespace {
struct t5 { };
enum LocalEnum { LocalEnum1 };
}
template<typename ...T1, typename T2 = int>
void f5() { }
template<typename T1, typename ...T2>
void f6() { }
struct t6 {
  template<typename T>
  void operator<<(int) {
  }
  template<typename T>
  void operator<(int) {
  }
  template<typename T>
  void operator<=(int) {
  }
  template<typename T = int>
  operator t1<float>*() {
    return nullptr;
  }
  template<typename T>
  void operator-(int) {
  }
  template<typename T>
  void operator*(int) {
  }
  template<typename T>
  void operator/(int) {
  }
  template<typename T>
  void operator%(int) {
  }
  template<typename T>
  void operator^(int) {
  }
  template<typename T>
  void operator&(int) {
  }
  template<typename T>
  void operator|(int) {
  }
  template<typename T>
  void operator~() {
  }
  template<typename T>
  void operator!() {
  }
  template<typename T>
  void operator=(int) {
  }
  template<typename T>
  void operator>(int) {
  }
  template<typename T>
  void operator,(int) {
  }
  template<typename T>
  void operator()() {
  }
  template<typename T>
  void operator[](int) {
  }
  template<typename T>
  void operator<=>(int) {
  }
  template<typename T>
  void* operator new(std::size_t, T) {
    __builtin_unreachable();
  }
  template<typename T>
  void operator delete(void*, T) {
  }
  template<typename T>
  void* operator new[](std::size_t, T) {
    __builtin_unreachable();
  }
  template<typename T>
  void operator delete[](void*, T) {
  }
  template<typename T>
  int operator co_await() { __builtin_unreachable(); }

};
void operator"" _suff(unsigned long long) {}
template<template<typename...> class T> void f7() { }
template<template<typename...> class T, typename T2> void f8() { }
template<typename T>
struct t7;
using t7i = t7<int>;
template<typename T>
struct
__attribute__((__preferred_name__(t7i)))
t7 {
};
struct t8 {
  void mem();
};
namespace ns {
inline namespace inl {
template<typename T> struct t9 { };
}
}
template<typename T>
void (*f9())() {
  return nullptr;
}
struct t10 {
  template<typename T = void>
  t10() { }
};

template<typename T>
void operator_not_really() {
}

template<typename T, T ...A>
struct t11 {
};

struct t12 {
  t11<LocalEnum, LocalEnum1> v1;
};

int main() {
  struct { } A;
  auto L = []{};
  f1<int>();
  f1<float>();
  f1<bool>();
  f1<double>();
  f1<long>();
  f1<short>();
  f1<unsigned>();
  f1<unsigned long long>();
  f1<long long>();
  f1<udt>();
  f1<ns::udt>();
  f1<ns::udt*>();
  f1<ns::inner::udt>();
  f1<t1<int>>();
  f1<int, float>();
  f1<int *>();
  f1<int &>();
  f1<int &&>();
  f1<const int>();
  f1<int[3]>();
  f1<void>();
  f1<outer_class::inner_class>();
  f1<unsigned long>();
  f2<true, 3>();
  f3<ns::Enumeration, ns::Enumerator3, (ns::Enumeration)2>();
  f3<ns::EnumerationClass, ns::EnumerationClass::Enumerator3, (ns::EnumerationClass)2>();
  f3<ns::EnumerationSmall, ns::kNeg>();
  f3<decltype(ns::AnonEnum1), ns::AnonEnum3, (decltype(ns::AnonEnum1))2>();
  f3<LocalEnum, LocalEnum1>();
  f3<int*, &i>();
  f3<int*, nullptr>();
  t4<3> v2;
  f3<unsigned long, 1>();
  f3<unsigned long long, 1>();
  f3<long, 1>();
  f3<unsigned int, 1>();
  f3<short, 1>();
  f3<unsigned char, (char)0>();
  f3<signed char, (char)0>();
  f3<unsigned short, 1, 2>();
  f3<char, 0, 1, 6, 7, 13, 14, 31, 32, 33, (char)127, (char)128>();
  f3<__int128, ((__int128)9223372036854775807) * 2>();
  f4<unsigned int>();
  f1<t3<int>>();
  f1<t3<t3<int>>>();
  f1<decltype(L)>();
  t3<decltype(L)> v1;
  f1<t3<t3<decltype(L)>>>();
  f1<int(float)>();
  f1<void(...)>();
  f1<void(int, ...)>();
  f1<const int &>();
  f1<const int *&>();
  f1<t5>();
  f1<decltype(nullptr)>();
  f1<long*, long*>();
  f1<long*, udt*>();
  f1<void *const>();
  f1<const void *const *>();
  f1<void()>();
  f1<void(*)()>();
  f1<decltype(&L)>();
  f1<decltype(A)>();
  f1<decltype(&A)>();
  f5<t1<int>>();
  f5<>();
  f6<t1<int>>();
  f1<>();
  f1<const void*, const void*>();
  f1<t1<int*>*>();
  f1<int *[]>();
  t6 v6;
  v6.operator<< <int>(1);
  v6.operator< <int>(1);
  v6.operator<= <int>(1);
  v6.operator t1<float>*();
  v6.operator- <int>(3);
  v6.operator* <int>(3);
  v6.operator/ <int>(3);
  v6.operator% <int>(3);
  v6.operator^ <int>(3);
  v6.operator& <int>(3);
  v6.operator| <int>(3);
  v6.operator~ <int>();
  v6.operator! <int>();
  v6.operator= <int>(3);
  v6.operator> <int>(3);
  v6.operator, <int>(3);
  v6.operator() <int>();
  v6.operator[] <int>(3);
  v6.operator<=> <int>(3);
  t6::operator new(0, 0);
  t6::operator new[](0, 0);
  t6::operator delete(nullptr, 0);
  t6::operator delete[](nullptr, 0);
  v6.operator co_await<int>();
  42_suff;
  struct t7 { };
  f1<t7>();
  f1<int(&)[3]>();
  f1<int(*)[3]>();
  f7<t1>();
  f8<t1, int>();
  using namespace ns;
  ttp_user<inner::ttp>();
  f1<int*, decltype(nullptr)*>();
  t7i x;
  f1<t7i>();
  f7<ns::inl::t9>();
  f1<_Atomic(int)>();
  f1<int, long, volatile char>();
  f1<__attribute__((__vector_size__(sizeof(int) * 2))) int>();
  f1<int *const volatile>();
  f1<const volatile void>();
  f1<t1<decltype(L)>>();
  t10 v3;
  f1<void (::udt::*)() const>();
  f1<void (::udt::*)() volatile &>();
  f1<void (::udt::*)() const volatile &&>();
  f9<int>();
  f1<void (*const)()>();
  f1<char const (&)[1]>();
  f1<void () const &>();
  f1<void () volatile &&>();
  f1<void () const volatile>();
  f1<int *const[1]>();
  f1<int *const(&)[1]>();
  f1<void (::udt::* const&)()>();
  f1<void (*(int))(float)>();
  f1<t1<int>[1]>();
  f1<void (*)() noexcept>();
  f1<void (decltype(A))>();
  struct t8 { decltype(A) m; };
  f1<void(t8, decltype(A))>();
  f1<void(t8)>();
  operator_not_really<int>();
  t12 v4;
}
void t8::mem() {
  struct t7 { };
  f1<t7>();
  f1<decltype(&t8::mem)>();
}
