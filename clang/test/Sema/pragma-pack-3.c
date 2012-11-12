// RUN: %clang_cc1 -triple i686-apple-darwin9 %s -fsyntax-only -verify
// expected-no-diagnostics

// Stack: [], Alignment: 8

#pragma pack(push, 1)
// Stack: [8], Alignment: 1

#pragma pack(push, 4)
// Stack: [8, 1], Alignment: 4

// Note that this differs from gcc; pack() in gcc appears to pop the
// top stack entry and resets the current alignment. This is both
// inconsistent with MSVC, and the gcc documentation. In other cases,
// for example changing this to pack(8), I don't even understand what gcc
// is doing.

#pragma pack()
// Stack: [8, 1], Alignment: 8

#pragma pack(pop)
// Stack: [8], Alignment: 1
struct s0 {
  char f0;
  short f1;
};
int a[sizeof(struct s0) == 3 ? 1 : -1];

#pragma pack(pop)
// Stack: [], Alignment: 8
struct s1 {
  char f0;
  short f1;
};
int b[sizeof(struct s1) == 4 ? 1 : -1];
