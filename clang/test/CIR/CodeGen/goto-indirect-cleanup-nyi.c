// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify %s

// An indirect goto with an active VLA cleanup is not implemented.
int vla(int n) {
  int a[n];
  void *p = &&done;
  // expected-error@+1 {{indirect goto with active cleanup}}
  goto *p;
done:
  return a[0];
}
