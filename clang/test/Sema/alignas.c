// RUN: %clang_cc1 -fsyntax-only -verify -std=c11 -Dalignof=__alignof %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c11 -Dalignof=_Alignof -DUSING_C11_SYNTAX %s

_Alignas(3) int align_illegal; //expected-error {{requested alignment is not a power of 2}}
_Alignas(int) char align_big;
_Alignas(1) int align_small; // expected-error {{requested alignment is less than minimum}}
_Alignas(1) unsigned _Alignas(8) int _Alignas(1) align_multiple;

struct align_member {
  _Alignas(8) int member;
  _Alignas(1) char bitfield : 1; // expected-error {{'_Alignas' attribute cannot be applied to a bit-field}}
};

typedef _Alignas(8) char align_typedef; // expected-error {{'_Alignas' attribute only applies to variables and fields}}

void f(_Alignas(1) char c) { // expected-error {{'_Alignas' attribute cannot be applied to a function parameter}}
  _Alignas(1) register char k; // expected-error {{'_Alignas' attribute cannot be applied to a variable with 'register' storage class}}
}

#ifdef USING_C11_SYNTAX
// expected-warning@+4{{'_Alignof' applied to an expression is a GNU extension}}
// expected-warning@+4{{'_Alignof' applied to an expression is a GNU extension}}
// expected-warning@+4{{'_Alignof' applied to an expression is a GNU extension}}
#endif
_Static_assert(alignof(align_big) == alignof(int), "k's alignment is wrong");
_Static_assert(alignof(align_small) == 1, "j's alignment is wrong");
_Static_assert(alignof(align_multiple) == 8, "l's alignment is wrong");
_Static_assert(alignof(struct align_member) == 8, "quuux's alignment is wrong");
_Static_assert(sizeof(struct align_member) == 8, "quuux's size is wrong");
