struct RAII {
  int i = 0;
  RAII() { i++; }
  ~RAII() { i--; }
};
void f() {
  RAII();
}
