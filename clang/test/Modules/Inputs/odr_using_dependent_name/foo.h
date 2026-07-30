template <class T>
struct bar {
  using Ty = int;
};
template <class T>
struct foo : public bar<T> {
  using typename bar<T>::Ty;
  void baz(Ty);
};
