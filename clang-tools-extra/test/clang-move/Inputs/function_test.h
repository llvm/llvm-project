void f();

inline int g() { return 0; }

template<typename T>
void h(T t) {}

template<>
void h(int t) {}

class A {
 public:
  void f();
};
