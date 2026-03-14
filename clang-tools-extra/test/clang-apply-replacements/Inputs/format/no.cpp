class C {};

void f() { // This comment necessary to prevent formatting as void f() { ... }
  C *a = new C();
  // CHECK: {{^\ \ auto\ a\ \=\ new\ C\(\);}}
}
