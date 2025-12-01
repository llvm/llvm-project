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

int redecl(const char *fmt, ...) __attribute__((modular_format(__first_impl, "__first", "one"), format(printf, 1, 2)));
int redecl(const char *fmt, ...) __attribute__((modular_format(__second_impl, "__second", "two", "three")));

// CHECK-LABEL: define dso_local void @test_redecl(
// CHECK:    {{.*}} = call i32 (ptr, ...) @redecl(ptr noundef @.str) #[[ATTR_REDECL:[0-9]+]]
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

// CHECK: attributes #[[ATTR]] = { "modular-format"="printf,1,2,__modular_printf,__printf,float" }
// CHECK: attributes #[[ATTR_REDECL]] = { "modular-format"="printf,1,2,__second_impl,__second,three,two" }
// CHECK: attributes #[[ATTR_ORDER]] = { "modular-format"="printf,1,2,__modular_printf,__printf,a,b" }
