// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

typedef typeof(sizeof(0)) size_t;

void bar(char *s);
void *memset(void *s, int c, size_t n);
void *memcpy(void *d, const void *s, size_t n);
void *memmove(void *d, const void *s, size_t n);

// CHECK: define{{.*}} void @foo1({{.*}}) #[[NO_NOBUILTIN:[0-9]+]]
// CHECK:   call void @bar
// CHECK:   call void @llvm.memset
// CHECK:   call void @llvm.memcpy
// CHECK:   call void @llvm.memmove
void foo1(char *s, char *d, size_t n) {
  bar(s);
  memset(s, 0, n);
  memcpy(d, s, n);
  memmove(d, s, n);
}

// CHECK: define{{.*}} void @foo2({{.*}}) #[[NOBUILTIN_MEMSET:[0-9]+]]
// CHECK:   call void @bar
// CHECK:   {{.*}}call {{.*}} @memset
// CHECK:   call void @llvm.memcpy
// CHECK:   call void @llvm.memmove
void foo2(char *s, char *d, size_t n) __attribute__((no_builtin("memset"))) {
  bar(s);
  memset(s, 1, n);
  memcpy(d, s, n);
  memmove(d, s, n);
}

// CHECK: define{{.*}} void @foo3({{.*}}) #[[NOBUILTIN_MEMSET_MEMCPY:[0-9]+]]
// CHECK:   call void @bar
// CHECK:   {{.*}}call {{.*}} @memset
// CHECK:   {{.*}}call {{.*}} @memcpy
// CHECK:   call void @llvm.memmove
void foo3(char *s, char *d, size_t n) __attribute__((no_builtin("memset", "memcpy"))) {
  bar(s);
  memset(s, 2, n);
  memcpy(d, s, n);
  memmove(d, s, n);
}

// CHECK: define{{.*}} void @foo4({{.*}}) #[[NOBUILTINS:[0-9]+]]
// CHECK:   call void @bar
// CHECK:   {{.*}}call {{.*}} @memset
// CHECK:   {{.*}}call {{.*}} @memcpy
// CHECK:   {{.*}}call {{.*}} @memmove
void foo4(char *s, char *d, size_t n) __attribute__((no_builtin)) {
  bar(s);
  memset(s, 2, n);
  memcpy(d, s, n);
  memmove(s, d, n);
}

// CHECK-NOT: attributes #[[NO_NOBUILTIN]] = {{{.*}}"no-builtin-memset"{{.*}}}
// CHECK-NOT: attributes #[[NO_NOBUILTIN]] = {{{.*}}"no-builtin-memcpy"{{.*}}"no-builtin-memset"{{.*}}}
// CHECK-NOT: attributes #[[NO_NOBUILTIN]] = {{{.*}}"no-builtins"{{.*}}}
// CHECK:     attributes #[[NOBUILTIN_MEMSET]] = {{{.*}}"no-builtin-memset"{{.*}}}
// CHECK:     attributes #[[NOBUILTIN_MEMSET_MEMCPY]] = {{{.*}}"no-builtin-memcpy"{{.*}}"no-builtin-memset"{{.*}}}
// CHECK:     attributes #[[NOBUILTINS]] = {{{.*}}"no-builtins"{{.*}}}
