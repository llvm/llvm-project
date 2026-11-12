class Bar {
public:
  template<typename T>
  void f() {
    static const T y = 0;
  }
};
