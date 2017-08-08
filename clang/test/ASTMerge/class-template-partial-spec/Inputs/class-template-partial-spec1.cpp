template<typename T, class P>
struct TwoOptionTemplate {};

template<typename T>
struct TwoOptionTemplate<T, char> {
  int member;
};


template<typename T>
struct TwoOptionTemplate<T, double> {
  float member;
};

template<typename T>
struct TwoOptionTemplate<T, T> {
  T** member;
};

TwoOptionTemplate<int, char> X0;
TwoOptionTemplate<int, float> X1;
TwoOptionTemplate<void *, wchar_t> X2;
TwoOptionTemplate<long, long> X3;
TwoOptionTemplate<float, float> X4;
TwoOptionTemplate<long, long> SingleSource;
TwoOptionTemplate<char, double> SecondDoubleSource;


template<int I, class C>
struct IntTemplateSpec {};

template<class C>
struct IntTemplateSpec<4, C> {
  C member;
};

template<int I>
struct IntTemplateSpec<I, void *> {
  int member;
  static constexpr int val = I;
};

template<int I>
struct IntTemplateSpec<I, double> {
  char member;
  static constexpr int val = I;
};

IntTemplateSpec<4, wchar_t> Y0;
IntTemplateSpec<5, void *> Y1;
IntTemplateSpec<1, long> Y2;
IntTemplateSpec<3, int> Y3;
//template<int I> constexpr int IntTemplateSpec<I, double>::val;
IntTemplateSpec<42, double> NumberSource;
static_assert(NumberSource.val == 42);

namespace One {
namespace Two {
  // Just an empty namespace to ensure we can deal with multiple namespace decls.
}
}


namespace One {
namespace Two {
namespace Three {

template<class T>
class Parent {};

} // namespace Three

} // namespace Two

template<typename T, typename X>
struct Child1: public Two::Three::Parent<unsigned> {
  char member;
};

template<class T>
struct Child1<T, One::Two::Three::Parent<T>> {
  T member;
};

} // namespace One

One::Child1<int, double> Z0Source;

// Test import of nested namespace specifiers
template<typename T>
struct Outer {
  template<typename U> class Inner0;
};

template<typename X>
template<typename Y>
class Outer<X>::Inner0 {
public:
  void f(X, Y);
  template<typename Z> struct Inner1;
};

template<typename X>
template<typename Y>
void Outer<X>::Inner0<Y>::f(X, Y) {}

template<typename X>
template<typename Y>
template<typename Z>
class Outer<X>::Inner0<Y>::Inner1 {
public:
  void f(Y, Z);
};

template<typename X>
template<typename Y>
template<typename Z>
void Outer<X>::Inner0<Y>::Inner1<Z>::f(Y, Z) {}
