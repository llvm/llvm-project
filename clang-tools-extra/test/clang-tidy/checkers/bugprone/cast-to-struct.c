// RUN: %check_clang_tidy %s bugprone-cast-to-struct %t

struct S1 {
  int a;
};

struct S2 {
  char a;
};

union U1 {
  int a;
  char b;
};

union U2 {
  struct S1 a;
  char b;
};

typedef struct S1 TyS1;
typedef struct S1 *TyPS1;

typedef union U1 *TyPU1;

typedef int int_t;
typedef int * int_ptr_t;

struct S1 *test_simple(short *p) {
  return (struct S1 *)p;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: casting a 'short *' pointer to a 'struct S1 *' pointer and accessing a field can lead to memory access errors or data corruption [bugprone-cast-to-struct]
  struct S1 *s;
  int i;
  s = (struct S1 *)&i;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: casting a 'int *' pointer to a 'struct S1 *' pointer and accessing a field can lead to memory access errors or data corruption [bugprone-cast-to-struct]
}

struct S1 *test_cast_from_void(void *p) {
  return (struct S1 *)p;
}

struct S1 *test_cast_from_struct(struct S2 *p) {
  return (struct S1 *)p;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: casting a 'struct S2 *' pointer to a 'struct S1 *' pointer and accessing a field can lead to memory access errors or data corruption [bugprone-cast-to-struct]
}

TyPS1 test_cast_from_similar(struct S1 *p) {
  return (TyPS1)p;
}

void test_typedef(short *p1, int_t *p2, int_ptr_t p3) {
  TyS1 *a = (TyS1 *)p1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: casting a 'short *' pointer to a 'TyS1 *' (aka 'struct S1 *') pointer and accessing a field can lead to memory access errors or data corruption [bugprone-cast-to-struct]
  TyPS1 b = (TyPS1)p1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: casting a 'short *' pointer to a 'TyPS1' (aka 'struct S1 *') pointer and accessing a field can lead to memory access errors or data corruption [bugprone-cast-to-struct]
  struct S1 *c = (struct S1 *)p2;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: casting a 'int_t *' (aka 'int *') pointer to a 'struct S1 *' pointer and accessing a field can lead to memory access errors or data corruption [bugprone-cast-to-struct]
  struct S1 *d = (struct S1 *)p3;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: casting a 'int_ptr_t' (aka 'int *') pointer to a 'struct S1 *' pointer and accessing a field can lead to memory access errors or data corruption [bugprone-cast-to-struct]
}

void test_union(short *p1, union U1 *p2, TyPU1 p3) {
  union U1 *a = (union U1 *)p1;
  struct S1 *b = (struct S1 *)p2;
  struct S1 *c = (struct S1 *)p3;
}
