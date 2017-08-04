// RUN: %clang_cc1 -verify -fblocks -cl-std=CL2.0 %s

// OpenCL v2.0 s6.12.5
void f0(int (^const bl)());
// All blocks declarations must be const qualified and initialized.
void f1() {
  int (^bl1)(void) = ^() {
    return 1;
  };
  int (^const bl2)(void) = ^() {
    return 1;
  };
  f0(bl1);
  f0(bl2);
  bl1 = bl2;          // expected-error{{invalid operands to binary expression ('int (__generic ^const)(void)' and 'int (__generic ^const)(void)')}}
  int (^const bl3)(); // expected-error{{invalid block variable declaration - must be initialized}}
}

// A block with extern storage class is not allowed.
extern int (^bl)(void) = ^() { // expected-error{{invalid block variable declaration - using 'extern' storage class is disallowed}}
  return 1;
};
void f2() {
  extern int (^bl)(void) = ^() { // expected-error{{invalid block variable declaration - using 'extern' storage class is disallowed}}
    return 1;
  };
}

// A block cannot be the return value of a function.
typedef int (^bl_t)(void);
bl_t f3(bl_t bl); // expected-error{{declaring function return value of type 'bl_t' (aka 'int (__generic ^const)(void)') is not allowed}}

struct bl_s {
  int (^bl)(void); // expected-error {{the 'int (__generic ^const)(void)' type cannot be used to declare a structure or union field}}
};

void f4() {
  __block int a = 10; // expected-error {{the __block storage type is not permitted}}
}

// A block with variadic argument is not allowed.
int (^bl)(int, ...) = ^int(int I, ...) { // expected-error {{invalid prototype, variadic arguments are not allowed in OpenCL}} expected-error {{invalid prototype, variadic arguments are not allowed in OpenCL}}
  return 0;
};
typedef int (^bl1_t)(int, ...); // expected-error {{invalid prototype, variadic arguments are not allowed in OpenCL}}

// A block can't be used to declare an array
typedef int (^bl2_t)(int);
void f5(int i) {
  bl2_t bl1 = ^(int i) {
    return 1;
  };
  bl2_t bl2 = ^(int i) {
    return 2;
  };
  bl2_t arr[] = {bl1, bl2}; // expected-error {{array of 'bl2_t' (aka 'int (__generic ^const)(int)') type is invalid in OpenCL}}
  int tmp = i ? bl1(i)      // expected-error {{block type cannot be used as expression in ternary expression in OpenCL}}
              : bl2(i);     // expected-error {{block type cannot be used as expression in ternary expression in OpenCL}}
}
// A block pointer type and all pointer operations are disallowed
void f6(bl2_t *bl_ptr) { // expected-error{{pointer to type '__generic bl2_t' (aka 'int (__generic ^const __generic)(int)') is invalid in OpenCL}}
  bl2_t bl = ^(int i) {
    return 1;
  };
  bl2_t *p; // expected-error {{pointer to type '__generic bl2_t' (aka 'int (__generic ^const __generic)(int)') is invalid in OpenCL}}
  *bl;      // expected-error {{invalid argument type 'bl2_t' (aka 'int (__generic ^const)(int)') to unary expression}}
  &bl;      // expected-error {{invalid argument type 'bl2_t' (aka 'int (__generic ^const)(int)') to unary expression}}
}
// A block can't reference another block
kernel void f7() {
  bl2_t bl1 = ^(int i) {
    return 1;
  };
  void (^bl2)(void) = ^{
    int i = bl1(1); // expected-error {{cannot refer to a block inside block}}
  };
  void (^bl3)(void) = ^{
  };
  void (^bl4)(void) = ^{
    bl3(); // expected-error {{cannot refer to a block inside block}}
  };
  return;
}
