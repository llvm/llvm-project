typedef int Int1;
using Int2 = int;

template<class T>
struct A {};

template <class T> using B = A<T>;

class C {
  typedef int Int3;
};
