// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm < %s | FileCheck %s

// CHECK: define void @foo(i32* nonnull %x)
void foo(int * __attribute__((nonnull)) x) {
  *x = 0;
}

// CHECK: define void @bar(i32* nonnull %x)
void bar(int * x) __attribute__((nonnull(1)))  {
  *x = 0;
}

// CHECK: define void @bar2(i32* %x, i32* nonnull %y)
void bar2(int * x, int * y) __attribute__((nonnull(2)))  {
  *x = 0;
}

static int a;
// CHECK: define nonnull i32* @bar3()
int * bar3() __attribute__((returns_nonnull))  {
  return &a;
}

// CHECK: define i32 @bar4(i32 %n, i32* nonnull %p)
int bar4(int n, int *p) __attribute__((nonnull)) {
  return n + *p;
}

// CHECK: define i32 @bar5(i32 %n, i32* nonnull %p)
int bar5(int n, int *p) __attribute__((nonnull(1, 2))) {
  return n + *p;
}

typedef union {
  unsigned long long n;
  int *p;
  double d;
} TransparentUnion __attribute__((transparent_union));

// CHECK: define i32 @bar6(i64 %
int bar6(TransparentUnion tu) __attribute__((nonnull(1))) {
  return *tu.p;
}

// CHECK: define void @bar7(i32* nonnull %a, i32* nonnull %b)
void bar7(int *a, int *b) __attribute__((nonnull(1)))
__attribute__((nonnull(2))) {}

// CHECK: define void @bar8(i32* nonnull %a, i32* nonnull %b)
void bar8(int *a, int *b) __attribute__((nonnull))
__attribute__((nonnull(1))) {}

// CHECK: declare void @foo_decl(i32* nonnull)
void foo_decl(int *__attribute__((nonnull)));

// CHECK: declare void @bar_decl(i32* nonnull)
void bar_decl(int *) __attribute__((nonnull(1)));

// CHECK: declare void @bar2_decl(i32*, i32* nonnull)
void bar2_decl(int *, int *) __attribute__((nonnull(2)));

// CHECK: declare nonnull i32* @bar3_decl()
int *bar3_decl(void) __attribute__((returns_nonnull));

// CHECK: declare i32 @bar4_decl(i32, i32* nonnull)
int bar4_decl(int, int *) __attribute__((nonnull));

// CHECK: declare i32 @bar5_decl(i32, i32* nonnull)
int bar5_decl(int, int *) __attribute__((nonnull(1, 2)));

// CHECK: declare i32 @bar6_decl(i64)
int bar6_decl(TransparentUnion) __attribute__((nonnull(1)));

// CHECK: declare void @bar7_decl(i32* nonnull, i32* nonnull)
void bar7_decl(int *, int *)
    __attribute__((nonnull(1))) __attribute__((nonnull(2)));

// CHECK: declare void @bar8_decl(i32* nonnull, i32* nonnull)
void bar8_decl(int *, int *)
    __attribute__((nonnull)) __attribute__((nonnull(1)));

// Clang specially disables nonnull attributes on some library builtin
// functions to work around the fact that the standard and some vendors mark
// them as nonnull even though they are frequently called in practice with null
// arguments if a corresponding size argument is zero.

// CHECK: declare i8* @memcpy(i8*, i8*, i64)
void *memcpy(void *, const void *, unsigned long)
    __attribute__((nonnull(1, 2))) __attribute__((returns_nonnull));

// CHECK: declare i32 @memcmp(i8*, i8*, i64)
int memcmp(const void *, const void *, unsigned long) __attribute__((nonnull(1, 2)));

// CHECK: declare i8* @memmove(i8*, i8*, i64)
void *memmove(void *, const void *, unsigned long)
    __attribute__((nonnull(1, 2))) __attribute__((returns_nonnull));

// CHECK: declare i8* @strncpy(i8*, i8*, i64)
char *strncpy(char *, const char *, unsigned long)
    __attribute__((nonnull(1, 2))) __attribute__((returns_nonnull));

// CHECK: declare i32 @strncmp(i8*, i8*, i64)
int strncmp(const char *, const char *, unsigned long) __attribute__((nonnull(1, 2)));

// CHECK: declare nonnull i8* @strncat(i8* nonnull, i8*, i64)
char *strncat(char *, const char *, unsigned long)
    __attribute__((nonnull(1, 2))) __attribute__((returns_nonnull));

// CHECK: declare i8* @memchr(i8*, i32, i64)
void *memchr(const void *__attribute__((nonnull)), int, unsigned long)
    __attribute__((returns_nonnull));

// CHECK: declare i8* @memset(i8*, i32, i64)
void *memset(void *__attribute__((nonnull)), int, unsigned long)
    __attribute__((returns_nonnull));

void use_declarations(int *p, void *volatile *sink) {
  foo_decl(p);
  bar_decl(p);
  bar2_decl(p, p);
  (void)bar3_decl();
  bar4_decl(42, p);
  bar5_decl(42, p);
  bar6_decl(p);
  bar7_decl(p, p);
  bar8_decl(p, p);
  *sink = (void *)&memcpy;
  *sink = (void *)&memcmp;
  *sink = (void *)&memmove;
  *sink = (void *)&strncpy;
  *sink = (void *)&strncmp;
  *sink = (void *)&strncat;
  *sink = (void *)&memchr;
  *sink = (void *)&memset;
}
