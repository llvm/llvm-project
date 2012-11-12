// RUN: %clang_cc1 -triple x86_64-unk-unk -o - -emit-llvm -g %s | FileCheck %s

// PR3023
void convert(void) {
  struct { typeof(0) f0; } v0;
}


// PR2784
struct OPAQUE; // CHECK: DW_TAG_structure_type
typedef struct OPAQUE *PTR;
PTR p;


// PR2950
struct s0;
struct s0 { struct s0 *p; } g0;

struct s0 *f0(struct s0 *a0) {
  return a0->p;
}


// PR3134
char xpto[];


// PR3427
struct foo {
  int a;
  void *ptrs[];
};
struct foo bar;


// PR4143
struct foo2 {
  enum bar *bar;
};

struct foo2 foo2;


// Radar 7325611
// CHECK: "barfoo"
typedef int barfoo;
barfoo foo() {
}

// CHECK: __uint128_t
__uint128_t foo128 ()
{
  __uint128_t int128 = 44;
  return int128;
}

// CHECK: uint64x2_t
typedef unsigned long long uint64_t;
typedef uint64_t uint64x2_t __attribute__((ext_vector_type(2)));
uint64x2_t extvectbar[4];
