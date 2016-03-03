// RUN: %clang_cc1 -verify -fblocks -cl-std=CL2.0 %s

// OpenCL v2.0 s6.12.5

// All blocks declarations must be const qualified and initialized.
void f1() {
  int (^bl1)() = ^() {}; // expected-error{{invalid block variable declaration - must be const qualified}}
  int (^const bl2)(); // expected-error{{invalid block variable declaration - must be initialized}}
  int (^const bl3)() = ^const(){
  };
}

// A block with extern storage class is not allowed.
extern int (^const bl)() = ^const(){}; // expected-error{{invalid block variable declaration - using 'extern' storage class is disallowed}}
void f2() {
  extern int (^const bl)() = ^const(){}; // expected-error{{invalid block variable declaration - using 'extern' storage class is disallowed}}
}

// A block cannot be the return value of a function.
typedef int (^const bl_t)(void);
bl_t f3(bl_t bl); // expected-error{{declaring function return value of type 'bl_t' (aka 'int (^const)(void)') is not allowed}}

struct bl_s {
  int (^const bl)(void); // expected-error {{the 'int (^const)(void)' type cannot be used to declare a structure or union field}}
};

void f4() {
  __block int a = 10; // expected-error {{the __block storage type is not permitted}}
}

// A block with variadic argument is not allowed.
int (^const bl)(int, ...) = ^const int(int I, ...) { // expected-error {{invalid block prototype, variadic arguments are not allowed in OpenCL}}
  return 0;
};

// A block can't be used to declare an array
typedef int (^const bl1_t)(int);
void f5(int i) {
  bl1_t bl1 = ^const(int i) {return 1;};
  bl1_t bl2 = ^const(int i) {return 2;};
  bl1_t arr[] = {bl1, bl2}; // expected-error {{array of 'bl1_t' (aka 'int (^const)(int)') type is invalid in OpenCL}}
  int tmp = i ? bl1(i)      // expected-error {{block type cannot be used as expression in ternary expression in OpenCL}}
              : bl2(i);     // expected-error {{block type cannot be used as expression in ternary expression in OpenCL}}
}

void f6(bl1_t * bl_ptr) {
  bl1_t bl = ^const(int i) {return 1;};
  bl1_t *p = &bl; // expected-error {{invalid argument type 'bl1_t' (aka 'int (^const)(int)') to unary expression}}
  bl = *bl_ptr;  // expected-error {{dereferencing pointer of type '__generic bl1_t *' (aka 'int (^const __generic *)(int)') is not allowed in OpenCL}}
}
