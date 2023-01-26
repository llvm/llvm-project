// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm < %s | FileCheck -check-prefix=NULL-INVALID %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -fno-delete-null-pointer-checks < %s | FileCheck -check-prefix=NULL-VALID %s

// NULL-INVALID: define{{.*}} void @foo(ptr noundef nonnull %x)
// NULL-VALID: define{{.*}} void @foo(ptr noundef %x)
void foo(int * __attribute__((nonnull)) x) {
  *x = 0;
}

// NULL-INVALID: define{{.*}} void @bar(ptr noundef nonnull %x)
// NULL-VALID: define{{.*}} void @bar(ptr noundef %x)
void bar(int * x) __attribute__((nonnull(1)))  {
  *x = 0;
}

// NULL-INVALID: define{{.*}} void @bar2(ptr noundef %x, ptr noundef nonnull %y)
// NULL-VALID: define{{.*}} void @bar2(ptr noundef %x, ptr noundef %y)
void bar2(int * x, int * y) __attribute__((nonnull(2)))  {
  *x = 0;
}

static int a;
// NULL-INVALID: define{{.*}} nonnull ptr @bar3()
// NULL-VALID: define{{.*}} ptr @bar3()
int * bar3(void) __attribute__((returns_nonnull))  {
  return &a;
}

// NULL-INVALID: define{{.*}} i32 @bar4(i32 noundef %n, ptr noundef nonnull %p)
// NULL-VALID: define{{.*}} i32 @bar4(i32 noundef %n, ptr noundef %p)
int bar4(int n, int *p) __attribute__((nonnull)) {
  return n + *p;
}

// NULL-INVALID: define{{.*}} i32 @bar5(i32 noundef %n, ptr noundef nonnull %p)
// NULL-VALID: define{{.*}} i32 @bar5(i32 noundef %n, ptr noundef %p)
int bar5(int n, int *p) __attribute__((nonnull(1, 2))) {
  return n + *p;
}

typedef union {
  unsigned long long n;
  int *p;
  double d;
} TransparentUnion __attribute__((transparent_union));

// NULL-INVALID: define{{.*}} i32 @bar6(i64 %
// NULL-VALID: define{{.*}} i32 @bar6(i64 %
int bar6(TransparentUnion tu) __attribute__((nonnull(1))) {
  return *tu.p;
}

// NULL-INVALID: define{{.*}} void @bar7(ptr noundef nonnull %a, ptr noundef nonnull %b)
// NULL-VALID: define{{.*}} void @bar7(ptr noundef %a, ptr noundef %b)
void bar7(int *a, int *b) __attribute__((nonnull(1)))
__attribute__((nonnull(2))) {}

// NULL-INVALID: define{{.*}} void @bar8(ptr noundef nonnull %a, ptr noundef nonnull %b)
// NULL-VALID: define{{.*}} void @bar8(ptr noundef %a, ptr noundef %b)
void bar8(int *a, int *b) __attribute__((nonnull))
__attribute__((nonnull(1))) {}
