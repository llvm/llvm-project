// RUN: %check_clang_tidy -std=c11-or-later %s bugprone-sizeof-expression %t

#define alignof(type_name) _Alignof(type_name)
extern void sink(const void *P);

enum { BufferSize = 1024 };

struct S {
  long A, B, C;
};

void bad4d(void) {
  struct S Buffer[BufferSize];

  struct S *P = &Buffer[0];
  struct S *Q = P;
  while (Q < P + alignof(Buffer)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: suspicious usage of 'alignof(...)' in pointer arithmetic; this scaled value will be scaled again by the '+' operator [bugprone-sizeof-expression]
    // CHECK-MESSAGES: :[[@LINE-2]]:16: note: '+' in pointer arithmetic internally scales with 'sizeof(struct S)' == {{[0-9]+}}
    sink(Q++);
  }
}
