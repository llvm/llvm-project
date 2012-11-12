// RUN: %clang_cc1 %s -fsyntax-only -verify

//typedef __attribute__(( ext_vector_type(4) ))  float float4;
typedef float float4 __attribute__((vector_size(16)));

float4 foo = (float4){ 1.0, 2.0, 3.0, 4.0 };

float4 foo2 = (float4){ 1.0, 2.0, 3.0, 4.0 , 5.0 }; // expected-warning{{excess elements in vector initializer}}

float4 array[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
int array_sizecheck[(sizeof(array) / sizeof(float4)) == 3 ? 1 : -1];

float4 array2[2] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 
                     9.0 }; // expected-warning {{excess elements in array initializer}}

float4 array3[2] = { {1.0, 2.0, 3.0}, 5.0, 6.0, 7.0, 8.0,
                     9.0 }; // expected-warning {{excess elements in array initializer}}

// PR5650
__attribute__((vector_size(16))) float f1(void) {
  __attribute__((vector_size(16))) float vec = {0.0f, 0.0f, 0.0f};
  return(vec);
}

__attribute__((vector_size(16))) float f2(
    __attribute__((vector_size(16))) float a1) {
  return(a1);
}



// PR5265
typedef float __attribute__((ext_vector_type (3))) float3;
int test2[sizeof(float3) == sizeof(float4) ? 1 : -1];

// rdar://problem/8345836
typedef long long __attribute__((vector_size(16))) longlong2;
typedef short __attribute__((vector_size(16))) short8;
typedef short __attribute__((vector_size(8))) short4;
void test3() {
  extern short8 test3_helper(void);
  longlong2 arr1[2] = { test3_helper(), test3_helper() };
  short4 arr2[2] = { test3_helper(), test3_helper() }; // expected-error 2 {{initializing 'short4' with an expression of incompatible type 'short8'}}
}
