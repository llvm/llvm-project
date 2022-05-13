// RUN: %clang_cc1 -emit-llvm -fms-extensions -o - %s | FileCheck %s

typedef typeof(sizeof(0)) size_t;

void bar(char *s);
void *memset(void *s, int c, size_t n);
void *memcpy(void *d, const void *s, size_t n);

// CHECK: define{{.*}} void @foo1({{.*}}) #[[NO_NOBUILTIN:[0-9]+]]
void foo1(char *s, char *d, size_t n) {
  bar(s);
  memset(s, 0, n);
  memcpy(d, s, n);
}

#pragma function(strlen, memset)

// CHECK: define{{.*}} void @foo2({{.*}}) #[[NOBUILTIN_MEMSET:[0-9]+]]
void foo2(char *s, char *d, size_t n) {
  bar(s);
  memset(s, 1, n);
  memcpy(d, s, n);
}

#pragma function(memcpy)

// CHECK: define{{.*}} void @foo3({{.*}}) #[[NOBUILTIN_MEMSET_MEMCPY:[0-9]+]]
void foo3(char *s, char *d, size_t n) {
  bar(s);
  memset(s, 2, n);
  memcpy(d, s, n);
}

// CHECK-NOT: attributes #[[NO_NOBUILTIN]] = {{{.*}}"no-builtin-memset"{{.*}}}
// CHECK-NOT: attributes #[[NO_NOBUILTIN]] = {{{.*}}"no-builtin-memcpy"{{.*}}"no-builtin-memset"{{.*}}}
// CHECK:     attributes #[[NOBUILTIN_MEMSET]] = {{{.*}}"no-builtin-memset"{{.*}}}
// CHECK-NOT: attributes #[[NOBUILTIN_MEMSET]] = {{{.*}}"no-builtin-memcpy"{{.*}}"no-builtin-memset"{{.*}}}
// CHECK:     attributes #[[NOBUILTIN_MEMSET_MEMCPY]] = {{{.*}}"no-builtin-memcpy"{{.*}}"no-builtin-memset"{{.*}}}
