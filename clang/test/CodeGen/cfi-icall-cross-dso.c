// RUN: %clang_cc1 -triple x86_64-unknown-linux -O1 \
// RUN:   -fsanitize=cfi-icall -fsanitize-cfi-cross-dso \
// RUN:   -emit-llvm -o - %s | FileCheck \
// RUN:       --check-prefix=CHECK --check-prefix=CHECK-DIAG \
// RUN:       --check-prefix=ITANIUM --check-prefix=ITANIUM-DIAG \
// RUN:       %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux -O1 \
// RUN:   -fsanitize=cfi-icall -fsanitize-cfi-cross-dso -fsanitize-trap=cfi-icall \
// RUN:   -emit-llvm -o - %s | FileCheck \
// RUN:       --check-prefix=CHECK \
// RUN:       --check-prefix=ITANIUM --check-prefix=ITANIUM-TRAP \
// RUN:       %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -O1 \
// RUN:   -fsanitize=cfi-icall -fsanitize-cfi-cross-dso \
// RUN:   -emit-llvm -o - %s | FileCheck \
// RUN:       --check-prefix=CHECK --check-prefix=CHECK-DIAG \
// RUN:       --check-prefix=MS --check-prefix=MS-DIAG \
// RUN:       %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -O1 \
// RUN:   -fsanitize=cfi-icall -fsanitize-cfi-cross-dso -fsanitize-trap=cfi-icall \
// RUN:   -emit-llvm -o - %s | FileCheck \
// RUN:       --check-prefix=CHECK \
// RUN:       --check-prefix=MS --check-prefix=MS-TRAP \
// RUN:       %s

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

// CHECK-DIAG: @[[SRC:.*]] = private unnamed_addr constant {{.*}}cfi-icall-cross-dso.c\00
// CHECK-DIAG: @[[TYPE:.*]] = private unnamed_addr constant { i16, i16, [{{.*}} x i8] } { i16 -1, i16 0, [{{.*}} x i8] c"'void ()'\00"
// CHECK-DIAG: @[[DATA:.*]] = private unnamed_addr global {{.*}}@[[SRC]]{{.*}}@[[TYPE]]


// ITANIUM: call i1 @llvm.bitset.test(i8* %{{.*}}, metadata !"_ZTSFvE"), !nosanitize
// ITANIUM-DIAG: call void @__cfi_slowpath_diag(i64 6588678392271548388, i8* %{{.*}}, {{.*}}@[[DATA]]{{.*}}) {{.*}}, !nosanitize
// ITANIUM-TRAP: call void @__cfi_slowpath(i64 6588678392271548388, i8* %{{.*}}) {{.*}}, !nosanitize

// MS: call i1 @llvm.bitset.test(i8* %{{.*}}, metadata !"?6AX@Z"), !nosanitize
// MS-DIAG: call void @__cfi_slowpath_diag(i64 4195979634929632483, i8* %{{.*}}, {{.*}}@[[DATA]]{{.*}}) {{.*}}, !nosanitize
// MS-TRAP: call void @__cfi_slowpath(i64 4195979634929632483, i8* %{{.*}}) {{.*}}, !nosanitize

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
