BEGIN
template<typename T = int> struct A {};
template<typename T> struct B {};
template<typename T> struct C;
template<typename T> struct D;
template<typename T> struct E;
template<typename T = int> struct G;
template<typename T = int> struct H;
template<typename T> struct J {};
template<typename T = int> struct J;
struct K : J<> {};
template<typename T = void> struct L;
struct FriendL {
  template<typename T> friend struct L;
};
END

namespace DeferredLookup {
  template<typename T, typename U = T> using X = U;
  template<typename T> void f() { (void) X<T>(); }
  template<typename T> int n = X<T>();
  template<typename T> struct S { X<T> xt; enum E : int; };
  template<typename T> enum S<T>::E : int { a = X<T>() };

  namespace Indirect {
    template<typename, bool = true> struct A {};
    template<typename> struct B { template<typename T> using C = A<T>; };
  }
}
