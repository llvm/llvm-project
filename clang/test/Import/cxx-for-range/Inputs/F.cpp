struct Container {
  int *begin();
  int *end();
};

void f() {
  for (Container c; int varname : c) {
    return;
  }
}
