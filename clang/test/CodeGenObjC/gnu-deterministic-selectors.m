// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -fobjc-runtime=gnustep-1.5 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -fobjc-runtime=gcc %s -emit-llvm -o - | FileCheck %s

// Check that these selectors are emitted in alphabetical order.
// The order doesn't actually matter, only that it doesn't vary across runs.
// Clang sorts them when targeting a GCC-like ABI to guarantee determinism.
// CHECK: @.objc_selector_list = internal global [6 x { ptr, ptr }] [{ ptr, ptr } { ptr @.objc_sel_namea, ptr null }, { ptr, ptr } { ptr @.objc_sel_nameg, ptr null }, { ptr, ptr } { ptr @.objc_sel_namej, ptr null }, { ptr, ptr } { ptr @.objc_sel_namel, ptr null }, { ptr, ptr } { ptr @.objc_sel_namez, ptr null }, { ptr, ptr } zeroinitializer], align 8


void f(void) {
	SEL a = @selector(z);
	SEL b = @selector(a);
	SEL c = @selector(g);
	SEL d = @selector(l);
	SEL e = @selector(j);
}
