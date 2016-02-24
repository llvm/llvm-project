// RUN: %clang_cc1 -verify -fblocks -cl-std=CL2.0 %s

int (^BlkVariadic)(int, ...) = ^int(int I, ...) { // expected-error {{invalid block prototype, variadic arguments are not allowed in OpenCL}}
  return 0;
};

typedef int (^BlkInt)(int);
void f1(int i) {
  BlkInt B1 = ^int(int I) {return 1;};
  BlkInt B2 = ^int(int I) {return 2;};
  BlkInt Arr[] = {B1, B2}; // expected-error {{array of 'BlkInt' (aka 'int (^)(int)') type is invalid in OpenCL}}
  int tmp = i ? B1(i)      // expected-error {{block type cannot be used as expression in ternary expression in OpenCL}}
              : B2(i);     // expected-error {{block type cannot be used as expression in ternary expression in OpenCL}}
}

void f2(BlkInt *BlockPtr) {
  BlkInt B = ^int(int I) {return 1;};
  BlkInt *P = &B; // expected-error {{invalid argument type 'BlkInt' (aka 'int (^)(int)') to unary expression}}
  B = *BlockPtr;  // expected-error {{dereferencing pointer of type '__generic BlkInt *' (aka 'int (^__generic *)(int)') is not allowed in OpenCL}}
}
