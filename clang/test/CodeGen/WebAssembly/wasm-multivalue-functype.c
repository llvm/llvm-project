// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +multivalue %s -S -O2 -o - | FileCheck %s

// CHECK: .functype f1 () -> ()
void f1(void) {}
// CHECK: .functype f1mv () -> ()
__attribute__((wasm_multivalue))
void f1mv(void) {}

// CHECK: .functype f2 (f32, f64) -> (i32)
int f2(float a, double b) { return (int)(a + b); }
// CHECK: .functype f2mv (f32, f64) -> (i32)
__attribute__((wasm_multivalue))
int f2mv(float a, double b) { return (int)(a + b); }

// CHECK: .functype f3 (i32, i64, i64) -> ()
__int128 f3(long double x) { return (__int128)x; }
// CHECK: .functype f3mv (i64, i64) -> (i64, i64)
__attribute__((wasm_multivalue))
__int128 f3mv(long double x) { return (__int128)x; }

struct Foo4 { };
union Bar4 { };

// CHECK: .functype f4 () -> ()
union Bar4 f4(struct Foo4 x) { union Bar4 r; return r; }
// CHECK: .functype f4mv () -> ()
__attribute__((wasm_multivalue))
union Bar4 f4mv(struct Foo4 x) { union Bar4 r; return r; }

struct Foo5 { int a; };
union Bar5 { int a; };

// CHECK: .functype f5 (i32) -> (i32)
union Bar5 f5(struct Foo5 x) { union Bar5 r; r.a = x.a; return r; }
// CHECK: .functype f5mv (i32) -> (i32)
__attribute__((wasm_multivalue))
union Bar5 f5mv(struct Foo5 x) { union Bar5 r; r.a = x.a; return r; }

struct Foo6 { int a; int b; };

// CHECK: .functype f6 (i32, i32) -> ()
struct Foo6 f6(struct Foo6 x) { return x; }
// CHECK: .functype f6mv (i32, i32) -> (i32, i32)
__attribute__((wasm_multivalue))
struct Foo6 f6mv(struct Foo6 x) { return x; }
