class Base1 {
public:
  void operator delete[](void *);
};
class Base2 {
public:
  void operator delete(void *);
};
struct Derived : Base1, Base2 {
  virtual ~Derived() {}
};
void in_h_tests(Derived *p, Derived *p1) {
  ::delete[] p;

  delete[] p1;
}
