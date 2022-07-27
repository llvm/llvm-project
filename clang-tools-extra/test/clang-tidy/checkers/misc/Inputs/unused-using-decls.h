class MyClass {
public:
  template <template <typename> class S, typename T>
  S<T> *func1(T *a) {
    return new S<T>();
  }
  template <typename T, T (*S)()>
  void func2(T a) {
    S();
  }
};
