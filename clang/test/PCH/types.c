// Test this without pch.
// RUN: %clang_cc1 -fblocks -include %S/types.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -fblocks -o %t %S/types.h
// RUN: %clang_cc1 -fblocks -include-pch %t -fsyntax-only -verify %s -ast-print

typedef int INT;
INT int_value;

__attribute__((address_space(1))) int int_as_one;

// TYPE_EXT_QUAL
ASInt *as_int_ptr1 = &int_value;  // expected-error{{changes address space of pointer}}
ASInt *as_int_ptr2 = &int_as_one;

// TYPE_COMPLEX
_Complex float Cfloat_val;
Cfloat *Cfloat_ptr = &Cfloat_val;

// TYPE_ATOMIC
_Atomic(int) AtomicInt_val;
AtomicInt *AtomicInt_ptr = &AtomicInt_val;

// TYPE_POINTER
int_ptr int_value_ptr = &int_value;

// TYPE_BLOCK_POINTER
void test_block_ptr(Block *bl) {
  *bl = ^(int x, float f) { return x; };
}

// TYPE_CONSTANT_ARRAY
five_ints fvi = { 1, 2, 3, 4, 5 };

// TYPE_INCOMPLETE_ARRAY
float_array fa1 = { 1, 2, 3 };
float_array fa2 = { 1, 2, 3, 4, 5, 6, 7, 8 };

// TYPE_VARIABLE_ARRAY in stmts.[ch]

// TYPE_VECTOR
float4 f4 = { 1.0, 2.0, 3.0, 4.0 };

// TYPE_EXT_VECTOR
ext_float4 ef4 = { 1.0, 2.0, 3.0, 4.0 };

// TYPE_FUNCTION_NO_PROTO
noproto np1;
int np1(x, y)
     int x;
     float y;
{
  return x;
}

// TYPE_FUNCTION_PROTO
proto p1;
float p1(float x, float y, ...) {
  return x + y;
}
proto *p2 = p1;

// TYPE_TYPEDEF
int_ptr_ptr ipp = &int_value_ptr;

// TYPE_TYPEOF_EXPR
typeof_17 *t17 = &int_value;
struct S { int x, y; };
typeof_17 t17_2 = (struct S){1, 2}; // expected-error{{initializing 'typeof_17' (aka 'int') with an expression of incompatible type 'struct S'}}

// TYPE_TYPEOF
int_ptr_ptr2 ipp2 = &int_value_ptr;
