// RUN: %clang_cc1 -triple x86_64-unknown-linux -O1 -fsanitize=cfi-icall -fsanitize-cfi-cross-dso -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=ITANIUM %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -O1 -fsanitize=cfi-icall  -fsanitize-cfi-cross-dso -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=MS %s

void caller(void (*f)()) {
  f();
}

static void g(void) {}
void h(void);

typedef void (*Fn)(void);
Fn g1() {
  return &g;
}
Fn h1() {
  return &h;
}

inline void foo() {}
void bar() { foo(); }

// ITANIUM: call i1 @llvm.bitset.test(i8* %{{.*}}, metadata !"_ZTSFvE"), !nosanitize
// ITANIUM: call void @__cfi_slowpath(i64 6588678392271548388, i8* %{{.*}}) {{.*}}, !nosanitize

// MS: call i1 @llvm.bitset.test(i8* %{{.*}}, metadata !"?6AX@Z"), !nosanitize
// MS: call void @__cfi_slowpath(i64 4195979634929632483, i8* %{{.*}}) {{.*}}, !nosanitize

// ITANIUM: define available_externally void @foo()
// MS: define linkonce_odr void @foo()

// Check that we emit both string and hash based bit set entries for static void g(),
// and don't emit them for the declaration of h().

// CHECK-NOT: !{!"{{.*}}", void ()* @h, i64 0}
// CHECK: !{!"{{.*}}", void ()* @g, i64 0}
// CHECK-NOT: !{!"{{.*}}", void ()* @h, i64 0}
// CHECK: !{i64 {{.*}}, void ()* @g, i64 0}
// CHECK-NOT: !{!"{{.*}}", void ()* @h, i64 0}

// ITANIUM-NOT: !{!{{.*}}, void ()* @foo,
// ITANIUM: !{!"_ZTSFvE", void ()* @bar, i64 0}
// ITANIUM-NOT: !{!{{.*}}, void ()* @foo,
// ITANIUM: !{i64 6588678392271548388, void ()* @bar, i64 0}
// ITANIUM-NOT: !{!{{.*}}, void ()* @foo,

// MS: !{!"?6AX@Z", void ()* @foo, i64 0}
// MS: !{i64 4195979634929632483, void ()* @foo, i64 0}

// CHECK: !{i32 4, !"Cross-DSO CFI", i32 1}
