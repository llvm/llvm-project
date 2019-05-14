class A {
public:
  struct { int foo; } f;
  struct { int foo; } g;
};

inline int useA(A &a) {
  return (a.f.foo + a.g.foo);
}
