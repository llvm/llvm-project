// RUN: %clang_cc1 -triple x86_64-unknown-linux %s -emit-llvm -funique-internal-linkage-names -o - | FileCheck %s

struct A;
static long foo(const struct A*p);

long bar(const struct A*p);
long bar(const struct A*p) __attribute__((__alias__("foo")));

// CHECK: ; Function Attrs: noinline nounwind optnone
// CHECK-NEXT: define internal i64 @foo(ptr noundef %0) #0 {
// CHECK-NEXT:   %2 = call i64 @_ZL3fooPK1A.__uniq.[[ATTR:[0-9]+]](ptr %0)
// CHECK-NEXT:   ret i64 %2
// CHECK-NEXT: }
// CHECK: define internal i64 @_ZL3fooPK1A.__uniq.[[ATTR:[0-9]+]](ptr noundef %p) #1 {
static long foo(const struct A*p) {return 1;}
