template<typename T, typename P>
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
TwoOptionTemplate<int, double> X1;
TwoOptionTemplate<void *, wchar_t> X2;
TwoOptionTemplate<long, long> X3;
TwoOptionTemplate<int, int> X4;
TwoOptionTemplate<long, long> SingleDest;
TwoOptionTemplate<int, double> SecondDoubleDest;


template<int I, class C>
struct IntTemplateSpec {};

template<class C>
struct IntTemplateSpec<4, C> {
  C member;
};

template<int I>
struct IntTemplateSpec<I, void *> {
  double member;
  static constexpr int val = I;
};

template<int I>
struct IntTemplateSpec<I, double> {
  char member;
  static constexpr int val = I;
};

IntTemplateSpec<4, wchar_t>Y0;
IntTemplateSpec<5, void *> Y1;
IntTemplateSpec<1, int> Y2;
IntTemplateSpec<2, int> Y3;
IntTemplateSpec<43, double> NumberDest;

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

namespace Dst { One::Child1<double, One::Two::Three::Parent<double>> Z0Dst; }
One::Child1<int, float> Z1;
