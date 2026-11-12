void f1(int &);
static void f2() {}
inline void foo() {
  static int i;
  f1(i);
  f2();
}
inline void foo2() {
}
void foo_ext() {}
