// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace pr12262 {

template<typename T, typename... Ts>
void abc1(int (*xxx)[sizeof ... (Ts) + 1]);

void qq1 () {
  abc1<int>(0);
  abc1<int,double>(0);
}


template <unsigned N> class array {};


template<typename T, typename... Types>
array<sizeof...(Types)> make_array1(Types&&... args);

void qq2 () {
  array<1> arr = make_array1<int>(1);
  array<3> arr2 = make_array1<int>(1,array<5>(),0.1);
}


template<typename T, typename... Types>
int make_array(array<sizeof...(Types)>&, Types... args);

void qq3 () {
  array<1> a1;
  int aa1 = make_array<int>(a1,1);
  array<2> a2;
  int aa2 = make_array<int>(a2, 0L, "abc");
}


template<typename ... Ts>
struct AAA {
  template<typename T, typename... Types>
  static array<sizeof...(Types)> make_array(Types ... args);
};

void qq4 () {
  array<2> arr2 = AAA<int, int>::make_array<int>(1,2);
}

}


namespace pr12439 {

template<class... Members>
struct X {
  template<int Idx>
  using get_t = decltype(sizeof...(Members));

  template<int i>
  get_t<i> get();
};

template<class... Members>
template<int i>
typename X<Members...>::template get_t<i> X<Members...>::get()
{
  return 0;
}

}


namespace pr13272 {

template<bool B, class T = void>
struct enable_if { };

template<class T> struct enable_if<true, T> {
  typedef T type;
};

class Exception {};

template<class Ex, typename... Args>
void cxx_throw(typename enable_if<(sizeof...(Args) > 0), const char *>::type fmt, Args&&... args) {
  return;
}

void test() {
  cxx_throw<Exception>("Youpi",1);
}

}


namespace pr13817 {

template <unsigned>
struct zod;

template <>
struct zod<1> {};

template <typename T, typename ... Ts>
zod<sizeof...(Ts)> make_zod(Ts ...) {
  return zod<sizeof...(Ts)>();
}

int main(int argc, char *argv[])
{
  make_zod<int>(1);
  return 0;
}

}


namespace pr14273 {

template<typename T, int i>
struct myType
{ };

template<typename T, typename... Args>
struct Counter
{
  static const int count = 1 + Counter<Args...>::count;
};

template<typename T>
struct Counter<T>
{
  static const int count = 1;
};

template<typename Arg, typename... Args>
myType<Arg, sizeof...(Args)>* make_array_with_type(const Args&... args)
{
  return 0;
}

void func(void)
{
  make_array_with_type<char>(1,2,3);
}

}


namespace pr15112
{
  template<bool, typename _Tp = void>
    struct enable_if
    { };
  template<typename _Tp>
    struct enable_if<true,_Tp>
    { typedef _Tp type; };

  typedef __typeof__(sizeof(int)) size_t;

  template <size_t n, typename T, typename... Args>
  struct is_array_of { static const bool value = true; };

  struct cpu { using value_type = void; };

  template <size_t Order, typename T>
  struct coords_alias { typedef T type; };

  template <size_t Order, typename MemoryTag>
  using coords = typename coords_alias<Order, MemoryTag>::type;

  template <typename MemTag, typename... Args>
  typename enable_if<is_array_of<sizeof...(Args), size_t, Args...>::value,
                     coords<sizeof...(Args), MemTag>>::type
    mkcoords(Args... args);

  auto c1 = mkcoords<cpu>(0ul, 0ul, 0ul);
}


namespace pr12699 {

template<bool B>
struct bool_constant
{
  static const bool value = B;
};

template<typename... A>
struct F
{
  template<typename... B>
    using SameSize = bool_constant<sizeof...(A) == sizeof...(B)>;

  template<typename... B, typename = SameSize<B...>>
  F(B...) { }
};

void func()
{
  F<int> f1(3);
}

}

#if __cplusplus >= 202002L
namespace GH81436 {

template <class E> struct Bar;

template <class E>
Bar(E) -> Bar<E>;

template <int> struct Foo {};

// Bar<Ts> doesn't have to be of a complete type.
template <class... Ts>
auto func() requires requires(Bar<Ts> ...init_lists) {
  sizeof...(init_lists) > 0;
} {}

void f() { func<int>(); }

} // namespace GH81436
#endif
