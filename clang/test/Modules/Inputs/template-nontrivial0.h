template <class T>
struct Class0 {
  Class0();
  Class0(const Class0<T> &);
  ~Class0();
  T *p;
};

struct S0 {
  id x;
};

Class0<S0> returnNonTrivial();
