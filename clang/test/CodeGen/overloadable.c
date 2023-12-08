// RUN: %clang_cc1 -triple %itanium_abi_triple -Wno-incompatible-function-pointer-types -emit-llvm %s -o - | FileCheck %s
// CHECK: _Z1fPA10_1X
// CHECK: _Z1fPFvvE

int __attribute__((overloadable)) f(int x) { return x; }
float __attribute__((overloadable)) f(float x) { return x; }
double __attribute__((overloadable)) f(double x) { return x; }
double _Complex __attribute__((overloadable)) f(double _Complex x) { return x; }
typedef short v4hi __attribute__ ((__vector_size__ (8)));
v4hi __attribute__((overloadable)) f(v4hi x) { return x; }

struct X { };
void  __attribute__((overloadable)) f(struct X (*ptr)[10]) { }

void __attribute__((overloadable)) f(int x, int y, ...) { }

void __attribute__((overloadable)) f(void (*x)(void)) {}

int main(void) {
  int iv = 17;
  float fv = 3.0f;
  double dv = 4.0;
  double _Complex cdv;
  v4hi vv;

  iv = f(iv);
  fv = f(fv);
  dv = f(dv);
  cdv = f(cdv);
  vv = f(vv);
}

// Ensuring that we pick the correct function for taking the address of an
// overload when conversions are involved.

void addrof_many(int *a) __attribute__((overloadable, enable_if(0, "")));
void addrof_many(void *a) __attribute__((overloadable));
void addrof_many(char *a) __attribute__((overloadable));

void addrof_single(int *a) __attribute__((overloadable, enable_if(0, "")));
void addrof_single(char *a) __attribute__((overloadable, enable_if(0, "")));
void addrof_single(char *a) __attribute__((overloadable));

// CHECK-LABEL: define {{(dso_local )?}}void @foo
void foo(void) {
  // CHECK: store ptr @_Z11addrof_manyPc
  void (*p1)(char *) = &addrof_many;
  // CHECK: store ptr @_Z11addrof_manyPv
  void (*p2)(void *) = &addrof_many;
  // CHECK: ptr @_Z11addrof_manyPc
  void *vp1 = (void (*)(char *)) & addrof_many;
  // CHECK: ptr @_Z11addrof_manyPv
  void *vp2 = (void (*)(void *)) & addrof_many;

  // CHECK: store ptr @_Z13addrof_singlePc
  void (*p3)(char *) = &addrof_single;
  // CHECK: @_Z13addrof_singlePc
  void (*p4)(int *) = &addrof_single;
  // CHECK: @_Z13addrof_singlePc
  void *vp3 = &addrof_single;
}


void ovl_bar(char *) __attribute__((overloadable));
void ovl_bar(int) __attribute__((overloadable));

// CHECK-LABEL: define {{(dso_local )?}}void @bar
void bar(void) {
  char charbuf[1];
  unsigned char ucharbuf[1];

  // CHECK: call void @_Z7ovl_barPc
  ovl_bar(charbuf);
  // CHECK: call void @_Z7ovl_barPc
  ovl_bar(ucharbuf);
}

void ovl_baz(int *, int) __attribute__((overloadable));
void ovl_baz(unsigned int *, unsigned int) __attribute__((overloadable));
void ovl_baz2(int, int *) __attribute__((overloadable));
void ovl_baz2(unsigned int, unsigned int *) __attribute__((overloadable));
// CHECK-LABEL: define {{(dso_local )?}}void @baz
void baz(void) {
  unsigned int j;
  // Initial rules for incompatible pointer conversions made this overload
  // ambiguous.
  // CHECK: call void @_Z7ovl_bazPjj
  ovl_baz(&j, 0);
  // CHECK: call void @_Z7ovl_bazPjj
  ovl_baz(&j, 0u);

  // CHECK: call void @_Z8ovl_baz2jPj
  ovl_baz2(0, &j);
  // CHECK: call void @_Z8ovl_baz2jPj
  ovl_baz2(0u, &j);
}
