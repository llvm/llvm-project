// RUN: %clang_cc1 -Wno-int-to-pointer-cast -Wno-pointer-to-int-cast -pedantic -fsyntax-only %s -verify -fblocks

typedef void (^CL)(void);

CL foo() {
  short y;
  short (^add1)(void) = ^{ return y+1; }; // expected-error {{incompatible block pointer types initializing 'short (^)(void)' with an expression of type 'int (^)(void)'}}

  CL X = ^{
    if (2)
      return;
    return 1;  // expected-error {{return type 'int' must match previous return type 'void' when block literal has unspecified explicit return type}}
  };

  int (^Y) (void)  = ^{
    if (3)
      return 1;
    else
      return; // expected-error {{return type 'void' must match previous return type 'int' when block literal has unspecified explicit return type}}
  };

  char *(^Z)(void) = ^{
    if (3)
      return "";
    else
      return (char*)0;
  };

  double (^A)(void) = ^ { // expected-error {{incompatible block pointer types initializing 'double (^)(void)' with an expression of type 'float (^)(void)'}}
    if (1)
      return (float)1.0;
    else
      if (2)
        return (double)2.0; // expected-error {{return type 'double' must match previous return type 'float' when block literal has unspecified explicit return type}}
    return 1; // expected-error {{return type 'int' must match previous return type 'float' when block literal has unspecified explicit return type}}
  };
  char *(^B)(void) = ^{
    if (3)
      return "";
    else
      return 2; // expected-error {{return type 'int' must match previous return type 'char *' when block literal has unspecified explicit return type}}
  };

  return ^{ return 1; }; // expected-error {{incompatible block pointer types returning 'int (^)(void)' from a function with result type 'CL' (aka 'void (^)(void)')}}
}

typedef int (^CL2)(void);

CL2 foo2() {
  return ^{ return 1; };
}

typedef unsigned int * uintptr_t;
typedef char Boolean;
typedef int CFBasicHash;

#define INVOKE_CALLBACK2(P, A, B) (P)(A, B)

typedef struct {
    Boolean (^isEqual)(const CFBasicHash *, uintptr_t stack_value_or_key1, uintptr_t stack_value_or_key2, Boolean is_key);
} CFBasicHashCallbacks;

int foo3() {
    CFBasicHashCallbacks cb;
    
    Boolean (*value_equal)(uintptr_t, uintptr_t) = 0;
            
    cb.isEqual = ^(const CFBasicHash *table, uintptr_t stack_value_or_key1, uintptr_t stack_value_or_key2, Boolean is_key) {
      return (Boolean)(uintptr_t)INVOKE_CALLBACK2(value_equal, (uintptr_t)stack_value_or_key1, (uintptr_t)stack_value_or_key2);
    };
}

static int funk(char *s) {
  if (^{} == ((void*)0))
    return 1;
  else 
    return 0;
}
void next();
void foo4() {
  int (^xx)(const char *s) = ^(char *s) { return 1; }; // expected-error {{incompatible block pointer types initializing 'int (^)(const char *)' with an expression of type 'int (^)(char *)'}}
  int (*yy)(const char *s) = funk; // expected-warning {{incompatible function pointer types initializing 'int (*)(const char *)' with an expression of type 'int (char *)'}}
  
  int (^nested)(char *s) = ^(char *str) { void (^nest)(void) = ^(void) { printf("%s\n", str); }; next(); return 1; }; // expected-warning{{implicitly declaring library function 'printf' with type 'int (const char *, ...)'}} \
  // expected-note{{include the header <stdio.h> or explicitly provide a declaration for 'printf'}}
}

typedef void (^bptr)(void);

bptr foo5(int j) {
  __block int i;
  if (j)
    return ^{ ^{ i=0; }(); };  // expected-error {{returning block that lives on the local stack}}
  return ^{ i=0; };  // expected-error {{returning block that lives on the local stack}}
  return (^{ i=0; });  // expected-error {{returning block that lives on the local stack}}
  return (void*)(^{ i=0; });  // expected-error {{returning block that lives on the local stack}}
}

int (*funcptr3[5])(long);
int sz8 = sizeof(^int (*[5])(long) {return funcptr3;}); // expected-error {{block cannot return array type}} expected-warning {{incompatible pointer to integer conversion}}
int sz9 = sizeof(^int(*())()[3]{ }); // expected-error {{function cannot return array type}}

void foo6() {
  int (^b)(int) __attribute__((noreturn));
  b = ^ (int i) __attribute__((noreturn)) { return 1; };  // expected-error {{block declared 'noreturn' should not return}}
  b(1);
  int (^c)(void) __attribute__((noreturn)) = ^ __attribute__((noreturn)) { return 100; }; // expected-error {{block declared 'noreturn' should not return}}
}


void foo7()
{
 const int (^BB) (void) = ^{ const int i = 1; return i; }; // OK - initializing 'const int (^)(void)' with an expression of type 'int (^)(void)'

 const int (^CC) (void)  = ^const int{ const int i = 1; return i; };


  int i;
  int (^FF) (void)  = ^{ return i; }; // OK
  int (^EE) (void)  = ^{ return i+1; }; // OK

  __block int j;
  int (^JJ) (void)  = ^{ return j; }; // OK
  int (^KK) (void)  = ^{ return j+1; }; // OK

  __block const int k;
  const int cint = 100;

  int (^MM) (void)  = ^{ return k; };
  int (^NN) (void)  = ^{ return cint; };
}

// rdar://11069896
void (^blk)(void) = ^{
    return (void)0; // expected-warning {{void block literal should not return void expression}}
};

// rdar://13463504
enum Test8 { T8_a, T8_b, T8_c };
void test8(void) {
  extern void test8_helper(int (^)(int));
  test8_helper(^(int flag) { if (flag) return T8_a; return T8_b; });
}
void test8b(void) {
  extern void test8_helper2(char (^)(int)); // expected-note {{here}}
  test8_helper2(^(int flag) { if (flag) return T8_a; return T8_b; }); // expected-error {{passing 'enum Test8 (^)(int)' to parameter of type 'char (^)(int)'}}
}
