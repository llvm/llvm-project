// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -fobjc-runtime=gnustep-1.5 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -fobjc-runtime=gcc %s -emit-llvm -o - | FileCheck %s

// Check that these selectors are emitted in alphabetical order.
// The order doesn't actually matter, only that it doesn't vary across runs.
// Clang sorts them when targeting a GCC-like ABI to guarantee determinism.
// CHECK: @.objc_selector_list = internal global [6 x { i8*, i8* }] [{ i8*, i8* } { i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.objc_sel_namea, i64 0, i64 0), i8* null }, { i8*, i8* } { i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.objc_sel_nameg, i64 0, i64 0), i8* null }, { i8*, i8* } { i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.objc_sel_namej, i64 0, i64 0), i8* null }, { i8*, i8* } { i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.objc_sel_namel, i64 0, i64 0), i8* null }, { i8*, i8* } { i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.objc_sel_namez, i64 0, i64 0), i8* null }, { i8*, i8* } zeroinitializer], align 8


void f(void) {
	SEL a = @selector(z);
	SEL b = @selector(a);
	SEL c = @selector(g);
	SEL d = @selector(l);
	SEL e = @selector(j);
}
