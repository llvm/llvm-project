// RUN: %clang_cc1 -fsyntax-only -verify %s -triple spir-unknown-unknown

#pragma OPENCL EXTENSION cl_khr_fp16 : enable


// Disallowed: parameters with type
// bool, half, size_t, ptrdiff_t, intptr_t, and uintptr_t
// or a struct / union with any of these types in them

// TODO: Ban int types, size_t, ptrdiff_t ...

kernel void bool_arg(bool x) { } // expected-error{{'bool' cannot be used as the type of a kernel parameter}}

kernel void half_arg(half x) { } // expected-error{{'half' cannot be used as the type of a kernel parameter}}

typedef struct ContainsBool // expected-note{{within field of type 'ContainsBool' declared here}}
{
  bool x; // expected-note{{field of illegal type 'bool' declared here}}
} ContainsBool;

kernel void bool_in_struct_arg(ContainsBool x) { } // expected-error{{'ContainsBool' (aka 'struct ContainsBool') cannot be used as the type of a kernel parameter}}



typedef struct FooImage2D // expected-note{{within field of type 'FooImage2D' declared here}}
{
  // TODO: Clean up needed - we don't really need to check for image, event, etc
  // as a note here any longer.
  // They are diagnosed as an error for all struct fields (OpenCL v1.2 s6.9b,r).
  image2d_t imageField; // expected-note{{field of illegal type '__read_only image2d_t' declared here}} expected-error{{the '__read_only image2d_t' type cannot be used to declare a structure or union field}}
} FooImage2D;

kernel void image_in_struct_arg(FooImage2D arg) { } // expected-error{{struct kernel parameters may not contain pointers}}

typedef struct Foo // expected-note{{within field of type 'Foo' declared here}}
{
  int* ptrField; // expected-note{{field of illegal pointer type 'int *' declared here}}
} Foo;

kernel void pointer_in_struct_arg(Foo arg) { } // expected-error{{struct kernel parameters may not contain pointers}}

typedef union FooUnion // expected-note{{within field of type 'FooUnion' declared here}}
{
  int* ptrField; // expected-note{{field of illegal pointer type 'int *' declared here}}
} FooUnion;

kernel void pointer_in_union_arg(FooUnion arg) { }// expected-error{{union kernel parameters may not contain pointers}}

typedef struct NestedPointer // expected-note 2 {{within field of type 'NestedPointer' declared here}}
{
  int x;
  struct InnerNestedPointer
  {
    int* ptrField; // expected-note 3 {{field of illegal pointer type 'int *' declared here}}
  } inner; // expected-note 3 {{within field of type 'struct InnerNestedPointer' declared here}}
} NestedPointer;

kernel void pointer_in_nested_struct_arg(NestedPointer arg) { }// expected-error{{struct kernel parameters may not contain pointers}}

struct NestedPointerComplex // expected-note{{within field of type 'NestedPointerComplex' declared here}}
{
  int foo;
  float bar;

  struct InnerNestedPointerComplex
  {
    int innerFoo;
    int* innerPtrField; // expected-note{{field of illegal pointer type 'int *' declared here}}
  } inner; // expected-note{{within field of type 'struct InnerNestedPointerComplex' declared here}}

  float y;
  float z[4];
};

kernel void pointer_in_nested_struct_arg_complex(struct NestedPointerComplex arg) { }// expected-error{{struct kernel parameters may not contain pointers}}

typedef struct NestedBool // expected-note 2 {{within field of type 'NestedBool' declared here}}
{
  int x;
  struct InnerNestedBool
  {
    bool boolField; // expected-note 2 {{field of illegal type 'bool' declared here}}
  } inner; // expected-note 2 {{within field of type 'struct InnerNestedBool' declared here}}
} NestedBool;

kernel void bool_in_nested_struct_arg(NestedBool arg) { } // expected-error{{'NestedBool' (aka 'struct NestedBool') cannot be used as the type of a kernel parameter}}

// Warning emitted again for argument used in other kernel
kernel void bool_in_nested_struct_arg_again(NestedBool arg) { } // expected-error{{'NestedBool' (aka 'struct NestedBool') cannot be used as the type of a kernel parameter}}


// Check for note with a struct not defined inside the struct
typedef struct NestedBool2Inner
{
  bool boolField; // expected-note{{field of illegal type 'bool' declared here}}
} NestedBool2Inner;

typedef struct NestedBool2 // expected-note{{within field of type 'NestedBool2' declared here}}
{
  int x;
  NestedBool2Inner inner; // expected-note{{within field of type 'NestedBool2Inner' (aka 'struct NestedBool2Inner') declared here}}
} NestedBool2;

kernel void bool_in_nested_struct_2_arg(NestedBool2 arg) { } // expected-error{{'NestedBool2' (aka 'struct NestedBool2') cannot be used as the type of a kernel parameter}}


struct InnerInner
{
  int* foo;
  bool x;
};

struct Valid
{
  float c;
  float d;
};

struct Inner
{
  struct Valid v;
  struct InnerInner a;
  struct Valid g;
  struct InnerInner b;
};

struct AlsoUser // expected-note{{within field of type 'AlsoUser' declared here}}
{
  float x;
  struct Valid valid1;
  struct Valid valid2;
  struct NestedPointer aaaa; // expected-note{{within field of type 'struct NestedPointer' declared here}}
};

kernel void pointer_in_nested_struct_arg_2(struct Valid valid, struct NestedPointer arg, struct AlsoUser also) { } // expected-error 2 {{struct kernel parameters may not contain pointers}}
