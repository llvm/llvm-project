// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

int printf(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float")));
int myprintf(const char *fmt, ...)  __attribute__((modular_format(__modular_printf, "__printf", "float"), format(printf, 1, 2)));

// CHECK-LABEL: define dso_local void @test_inferred_format(
// CHECK:    {{.*}} = call i32 (ptr, ...) @printf(ptr noundef @.str) #[[ATTR:[0-9]+]]
void test_inferred_format(void) {
  printf("hello");
}

// CHECK-LABEL: define dso_local void @test_explicit_format(
// CHECK:    {{.*}} = call i32 (ptr, ...) @myprintf(ptr noundef @.str) #[[ATTR:[0-9]+]]
void test_explicit_format(void) {
  myprintf("hello");
}

int redecl(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
int redecl(const char *fmt, ...) __attribute__((modular_format(__dupe_impl, "__dupe", "1")));
int redecl(const char *fmt, ...) __attribute__((modular_format(__dupe_impl, "__dupe", "1")));

// CHECK-LABEL: define dso_local void @test_redecl(
// CHECK:    {{.*}} = call i32 (ptr, ...) @redecl(ptr noundef @.str) #[[ATTR_DUPE_IDENTICAL:[0-9]+]]
void test_redecl(void) {
  redecl("hello");
}

int order1(const char *fmt, ...) __attribute__((modular_format(__modular_printf, "__printf", "a", "b"), format(printf, 1, 2)));
int order2(const char *fmt, ...) __attribute__((modular_format(__modular_printf, "__printf", "b", "a"), format(printf, 1, 2)));

// CHECK-LABEL: define dso_local void @test_order(
// CHECK:    {{.*}} = call i32 (ptr, ...) @order1(ptr noundef @.str) #[[ATTR_ORDER:[0-9]+]]
// CHECK:    {{.*}} = call i32 (ptr, ...) @order2(ptr noundef @.str) #[[ATTR_ORDER]]
void test_order(void) {
  order1("hello");
  order2("hello");
}

int duplicate_identical(const char *fmt, ...) __attribute__((modular_format(__dupe_impl, "__dupe", "1"), modular_format(__dupe_impl, "__dupe", "1"), format(printf, 1, 2)));

// CHECK-LABEL: define dso_local void @test_duplicate_identical(
// CHECK:    {{.*}} = call i32 (ptr, ...) @duplicate_identical(ptr noundef @.str) #[[ATTR_DUPE_IDENTICAL]]
void test_duplicate_identical(void) {
  duplicate_identical("hello");
}

// CHECK: attributes #[[ATTR]] = { "modular-format"="printf,1,2,__modular_printf,__printf,float" }
// CHECK: attributes #[[ATTR_DUPE_IDENTICAL]] = { "modular-format"="printf,1,2,__dupe_impl,__dupe,1" }
// CHECK: attributes #[[ATTR_ORDER]] = { "modular-format"="printf,1,2,__modular_printf,__printf,a,b" }
