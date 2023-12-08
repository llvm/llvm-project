class MyType012345678901234567890123456789 {};

void g(int, int*, int, int*, int, int*, int);

void f() {
  MyType012345678901234567890123456789 *a =
      new MyType012345678901234567890123456789();
  // CHECK: {{^\ \ auto\ a\ \=\ new\ MyType012345678901234567890123456789\(\);}}

  int iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii;
  int jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj;
  int kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk;
  int mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm;
  g(iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii, 0, jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj,
    0, kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk, 0, mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm);
  // CHECK: g(iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii, nullptr,
  // CHECK-NEXT: jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj, nullptr,
  // CHECK-NEXT: kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk, nullptr,
  // CHECK-NEXT: mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm);

  delete a;
}
