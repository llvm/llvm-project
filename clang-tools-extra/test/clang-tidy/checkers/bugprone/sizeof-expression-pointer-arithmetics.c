// RUN: %check_clang_tidy %s bugprone-sizeof-expression %t

#define offsetof(type, member) __builtin_offsetof(type, member)

typedef __SIZE_TYPE__ size_t;
typedef __WCHAR_TYPE__ wchar_t;

extern void *memset(void *Dest, int Ch, size_t Count);
extern size_t strlen(const char *Str);
extern size_t wcslen(const wchar_t *Str);
extern char *strcpy(char *Dest, const char *Src);
extern wchar_t *wcscpy(wchar_t *Dest, const wchar_t *Src);
extern int scanf(const char *Format, ...);
extern int wscanf(const wchar_t *Format, ...);

extern void sink(const void *P);

enum { BufferSize = 1024 };

void bad1a(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  int *Q = P;
  while (Q < P + sizeof(Buffer)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic; this scaled value will be scaled again by the '+' operator [bugprone-sizeof-expression]
    // CHECK-MESSAGES: :[[@LINE-2]]:16: note: '+' in pointer arithmetic internally scales with 'sizeof(int)' == {{[0-9]+}}
    *Q++ = 0;
  }
}

void bad1b(void) {
  typedef int Integer;
  Integer Buffer[BufferSize];

  Integer *P = &Buffer[0];
  Integer *Q = P;
  while (Q < P + sizeof(Buffer)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic; this scaled value will be scaled again by the '+' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:16: note: '+' in pointer arithmetic internally scales with 'sizeof(Integer)' == {{[0-9]+}}
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
    // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic; this scaled value will be scaled again by the '+' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:21: note: '+' in pointer arithmetic internally scales with 'sizeof(int)' == {{[0-9]+}}
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
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: suspicious usage of 'offsetof(...)' in pointer arithmetic; this scaled value will be scaled again by the '+' operator
  // CHECK-MESSAGES: :[[@LINE-3]]:12: note: '+' in pointer arithmetic internally scales with 'sizeof(struct S)' == {{[0-9]+}}
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
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic; this scaled value will be scaled again by the '+' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:16: note: '+' in pointer arithmetic internally scales with 'sizeof(struct S)' == {{[0-9]+}}
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
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic; this scaled value will be scaled again by the '+=' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:7: note: '+=' in pointer arithmetic internally scales with 'sizeof(int)' == {{[0-9]+}}
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
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic; this scaled value will be scaled again by the '+' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:11: note: '+' in pointer arithmetic internally scales with 'sizeof(int)' == {{[0-9]+}}
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
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic; this scaled value will be scaled again by the '-' operator
    // CHECK-MESSAGES: :[[@LINE-2]]:11: note: '-' in pointer arithmetic internally scales with 'sizeof(int)' == {{[0-9]+}}
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

void bad12(void) {
  wchar_t Message[BufferSize];
  wcscpy(Message, L"Message: ");
  wscanf(L"%s", Message + wcslen(Message) * sizeof(wchar_t));
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: suspicious usage of 'sizeof(...)' in pointer arithmetic; this scaled value will be scaled again by the '+' operator
  // CHECK-MESSAGES: :[[@LINE-2]]:25: note: '+' in pointer arithmetic internally scales with 'sizeof(wchar_t)' == {{[0-9]+}}
}

void silenced12(void) {
  char Message[BufferSize];
  strcpy(Message, "Message: ");
  scanf("%s", Message + strlen(Message) * sizeof(char));
}

void nomatch12(void) {
  char Message[BufferSize];
  strcpy(Message, "Message: ");
  scanf("%s", Message + strlen(Message));
}

void good12(void) {
  wchar_t Message[BufferSize];
  wcscpy(Message, L"Message: ");
  wscanf(L"%s", Message + wcslen(Message));
}

void good13(void) {
  int Buffer[BufferSize];

  int *P = &Buffer[0];
  while (P < (Buffer + sizeof(Buffer) / sizeof(int))) {
    // NO-WARNING: Calculating the element count of the buffer here, which is
    // safe with this idiom (as long as the types don't change).
    ++P;
  }

  while (P < (Buffer + sizeof(Buffer) / sizeof(Buffer[0]))) {
    // NO-WARNING: Calculating the element count of the buffer here, which is
    // safe with this idiom.
    ++P;
  }

  while (P < (Buffer + sizeof(Buffer) / sizeof(*P))) {
    // NO-WARNING: Calculating the element count of the buffer here, which is
    // safe with this idiom.
    ++P;
  }
}
