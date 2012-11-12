// RUN: %clang_cc1 -fsyntax-only -verify -std=c11 -Dalignof=__alignof %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c11 -Dalignof=_Alignof %s

_Alignas(3) int align_illegal; //expected-error {{requested alignment is not a power of 2}}
_Alignas(int) char align_big;
_Alignas(1) int align_small; // FIXME: this should be rejected
_Alignas(1) unsigned _Alignas(8) int _Alignas(1) align_multiple;

struct align_member {
  _Alignas(8) int member;
};

typedef _Alignas(8) char align_typedef; // FIXME: this should be rejected

_Static_assert(alignof(align_big) == alignof(int), "k's alignment is wrong");
_Static_assert(alignof(align_small) == 1, "j's alignment is wrong");
_Static_assert(alignof(align_multiple) == 8, "l's alignment is wrong");
_Static_assert(alignof(struct align_member) == 8, "quuux's alignment is wrong");
_Static_assert(sizeof(struct align_member) == 8, "quuux's size is wrong");
_Static_assert(alignof(align_typedef) == 8, "typedef's alignment is wrong");
