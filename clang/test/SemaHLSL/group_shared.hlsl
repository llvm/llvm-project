// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

 groupshared float a[10];

 [numthreads(8,8,1)]
 void main() {
   a[0] = 1;
   // expected-error@+1 {{automatic variable qualified with an address space}}
   groupshared float b;
 }

// expected-warning@+2 {{'groupshared' attribute only applies to variables}}
// expected-error@+1 {{return type cannot be qualified with address space}}
 groupshared float foo() {
  static groupshared float foo0;
    return 1;
 }
// expected-warning@+2 {{'groupshared' attribute only applies to variables}}
// expected-error@+1 {{return type cannot be qualified with address space}}
  groupshared void bar() {
    extern groupshared float bar0;
  }
// expected-warning@+2 {{'groupshared' attribute only applies to variables}}
// expected-error@+1 {{return type cannot be qualified with address space}}
  groupshared float decl() {
      return 1;
  }

  class C {
      // expected-warning@+2 {{'groupshared' attribute only applies to variables}}
      // expected-error@+1 {{return type cannot be qualified with address space}}
      groupshared void foo() {}
      // expected-warning@+2 {{'groupshared' attribute only applies to variables}}
      // expected-error@+1 {{return type cannot be qualified with address space}}
      groupshared void bar();
      // expected-warning@+2 {{'groupshared' attribute only applies to variables}}
      // expected-error@+1 {{return type cannot be qualified with address space}}
      friend groupshared void friend_def() {}
      // expected-warning@+2 {{'groupshared' attribute only applies to variables}}
      // expected-error@+1 {{return type cannot be qualified with address space}}
      friend groupshared void friend_decl();
  };

  struct S {
    // expected-warning@+2 {{'groupshared' attribute only applies to variables}}
    // expected-error@+1 {{field may not be qualified with an address space}}
    groupshared float f;
    static groupshared float g;
  };

  // expected-error@+1 {{parameter may not be qualified with an address space}}
  float foo2(groupshared float a) {
    return a;
  }

// expected-note@+2 {{parameter may not be qualified with an address space}}
template<typename T>
  T tfoo(T t) {
     return t;
  }
  // expected-warning@+1 {{alias declarations are a C++11 extension}}
 using GSF = groupshared float;
 GSF gs;
 // expected-error@+1 {{no matching function for call to 'tfoo'}}
 GSF gs2 = tfoo<GSF>(gs);

// NOTE:This one didn't report error on the groupshared return type,
// it is caused by return type check is after pointer check which is acceptable.
// expected-error@+1 {{pointers are unsupported in HLSL}}
groupshared void (*fp)();
// expected-error@+2 {{pointers are unsupported in HLSL}}
// expected-error@+1 {{parameter may not be qualified with an address space}}
void (*fp2)(groupshared float);
// NOTE: HLSL not support trailing return types.
// expected-warning@+2 {{'auto' type specifier is a C++11 extension}}
// expected-error@+1 {{expected function body after function declarator}}
auto func() -> groupshared void;
// expected-warning@+2 {{'groupshared' attribute only applies to variables}}
// expected-error@+1 {{return type cannot be qualified with address space}}
void groupshared f();

struct S2 {
  // Do we reject it as a function qualifier on a member function?
  void f() groupshared;
};

// Does it impact size or alignment?
_Static_assert(sizeof(float) == sizeof(groupshared float), "");
_Static_assert(_Alignof(double) == _Alignof(groupshared double),"");

// Does it impact type identity for templates?
template <typename Ty>
struct S3 {
  static const bool value = false;
};

template <>
struct S3<groupshared float> {
  static const bool value = true;
};
_Static_assert(!S3<float>::value, "");
_Static_assert(S3<groupshared float>::value, "");

// Can you overload based on the qualifier?
void func(float f) {}
// expected-error@+1 {{parameter may not be qualified with an address space}}
void func(groupshared float f) {}
