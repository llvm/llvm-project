void f() {
  void const *l1_ptr = &&l1;
  goto *l1_ptr;
l1:
  return;
}
