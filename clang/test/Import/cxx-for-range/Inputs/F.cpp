struct Container {
  int *begin();
  int *end();
};

void f() {
  Container c;
  for (int varname : c) {
    return;
  }
}
