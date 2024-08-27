// RUN: %check_clang_tidy %s bugprone-suspicious-pointer-arithmetics-using-sizeof %t

typedef __typeof__(sizeof(void*)) size_t;
#define offsetof(type, member) __builtin_offsetof(type, member)
extern void *memset(void *Dst, int Ch, size_t Count);
extern void sink(const void *P);

enum { BufferSize = 1024 };

void bad1a(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = P;
  while (Q < P + sizeof(Buffer)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: pointer arithmetic using a number scaled by 'sizeof()'; this distance will be scaled again by the '+' operator [bugprone-suspicious-pointer-arithemetics-using-sizeof]
    // CHECK-MESSAGES: :[[@LINE-2]]:16: note: '+' scales with 'sizeof(Buffer)' == {{[0-9]+}}
    *Q++ = 0;
  }
}

void bad1b(void) {
  typedef int Integer;
  Integer Buffer[BufferSize];

  Integer *P = &Buffer[0];
  Integer *Q = P;
  while (Q < P + sizeof(Buffer)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: pointer arithmetic using a number scaled by 'sizeof()'; this distance will be scaled again by the '+' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:16: note: '+' scales with 'sizeof(Buffer)' == {{[0-9]+}}
    *Q++ = 0;
  }
}

void good1(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = P;
  while (Q < P + BufferSize) {
    *Q++ = 0;
  }
}

void bad2(void) {
  int Buffer[BufferSize];
  int *P = Buffer;

  while (P < Buffer + sizeof(Buffer)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: pointer arithmetic using a number scaled by 'sizeof()'; this value will be scaled again by the '+' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:16: note: '+' scales with 'sizeof(int)' == {{[0-9]+}}
    *P++ = 0;
  }
}

void good2(void) {
  int Buffer[BufferSize];
  int *P = Buffer;

  while (P < Buffer + BufferSize) {
    *P++ = 0;
  }
}

struct S {
  long A, B, C;
};

void bad3a(struct S *S) {
  const size_t Offset = offsetof(struct S, B);
  struct S *P = S;

  // This is not captureable by Tidy because the size/offset expression is
  // not a direct child of the pointer arithmetics.
  memset(P + Offset, 0, sizeof(struct S) - Offset);
}

void good3a(struct S *S) {
  const size_t Offset = offsetof(struct S, B);
  char *P = (char*)S;

  // This is not captureable by Tidy because the size/offset expression is
  // not a direct child of the pointer arithmetics.
  memset(P + Offset, 0, sizeof(struct S) - Offset);
}

void bad3b(struct S *S) {
  memset(S + offsetof(struct S, B), 0,
         sizeof(struct S) - offsetof(struct S, B));
  // NCHECK-MESSAGES: :[[@LINE-1]]:12: warning: pointer arithmetic using a number scaled by 'offsetof()'; this value will be scaled again by the '+' operator
  // NCHECK-MESSAGES: :[[@LINE-2]]:12: note: '+' scales with 'sizeof(int)' == {{[0-9]+}}
}

void good3b(struct S *S) {
  char *P = (char*)S;
  memset(P + offsetof(struct S, B), 0,
         sizeof(struct S) - offsetof(struct S, B));
}

void bad3c(void) {
  struct S Buffer[BufferSize];

  struct S *P = &Buffer[0];
  struct S *Q = P;
  while (Q < P + sizeof(Buffer)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: pointer arithmetic using a number scaled by 'sizeof()'; this value will be scaled again by the '+' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:16: note: '+' scales with 'sizeof(struct S)' == {{[0-9]+}}
    sink(Q++);
  }
}

void bad4(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = P;
  while (Q < P + BufferSize) {
    *Q = 0;
    Q += sizeof(*Q);
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: pointer arithmetic using a number scaled by 'sizeof()'; this value will be scaled again by the '+=' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:7: note: '+=' scales with 'sizeof(int)' == {{[0-9]+}}
  }
}

void silenced4(void) {
  char Buffer[BufferSize];

  char *P = &Buffer[0];
  char *Q = P;
  while (Q < P + BufferSize) {
    *Q = 0;
    Q += sizeof(*Q);
  }
}

void good4(void) {
  char Buffer[BufferSize];

  char *P = &Buffer[0];
  char *Q = P;
  while (Q < P + BufferSize) {
    *Q = 0;
    Q += 1;
  }
}

void good5aa(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = P;
  while (Q < P + BufferSize) {
    *Q = 0;
    Q += ( sizeof(Buffer) / sizeof(Buffer[0]) );
  }
}

void good5ab(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = P;
  while (Q < P + BufferSize) {
    *Q = 0;
    Q = Q + ( sizeof(Buffer) / sizeof(Buffer[0]) );
  }
}

void good5ba(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = P;
  while (Q < P + BufferSize) {
    *Q = 0;
    Q -= ( sizeof(Buffer) / sizeof(Buffer[0]) );
  }
}

void good5bb(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = P;
  while (Q < P + BufferSize) {
    *Q = 0;
    Q = Q - ( sizeof(Buffer) / sizeof(Buffer[0]) );
  }
}

void bad6(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = P;
  while (Q < P + BufferSize) {
    *Q = 0;
    Q = Q + sizeof(*Q);
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: pointer arithmetic using a number scaled by 'sizeof()'; this value will be scaled again by the '+' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:11: note: '+' scales with 'sizeof(int)' == {{[0-9]+}}
  }
}

void silenced6(void) {
  char Buffer[BufferSize];

  char *P = &Buffer[0];
  char *Q = P;
  while (Q < P + BufferSize) {
    *Q = 0;
    Q = Q + sizeof(*Q);
  }
}

void good6(void) {
  char Buffer[BufferSize];

  char *P = &Buffer[0];
  char *Q = P;
  while (Q < P + BufferSize) {
    *Q = 0;
    Q = Q + 1;
  }
}

void silenced7(void) {
  char Buffer[BufferSize];

  char *P = &Buffer[0];
  const char *Q = P;
  while (Q < P + BufferSize) {
    sink(Q);
    Q = Q + sizeof(*Q);
  }
}

void good7(void) {
  char Buffer[BufferSize];

  char *P = &Buffer[0];
  const char *Q = P;
  while (Q < P + BufferSize) {
    sink(Q);
    Q = Q + 1;
  }
}

void bad8(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = P;
  while (Q >= P) {
    *Q = 0;
    Q = Q - sizeof(*Q);
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: pointer arithmetic using a number scaled by 'sizeof()'; this value will be scaled again by the '-' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:11: note: '-' scales with 'sizeof(int)' == {{[0-9]+}}
  }
}

void silenced8(void) {
  char Buffer[BufferSize];

  char *P = &Buffer[0];
  char *Q = P;
  while (Q >= P) {
    *Q = 0;
    Q = Q - sizeof(*Q);
  }
}

void good8(void) {
  char Buffer[BufferSize];

  char *P = &Buffer[0];
  char *Q = P;
  while (Q >= P) {
    *Q = 0;
    Q = Q - 1;
  }
}

void good9(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = P + BufferSize;
  int N = Q - P;
  while (N >= 0) {
    Q[N] = 0;
    N = N - 1;
  }
}

void good10(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = Buffer + BufferSize;
  int I = sizeof(*P) - sizeof(*Q);

  sink(&I);
}

void good11(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = Buffer + BufferSize;
  int I = sizeof(Q) - sizeof(*P);

  sink(&I);
}
