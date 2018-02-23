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

// CHECK-DIAG: @[[SRC:.*]] = private unnamed_addr constant {{.*}}cfi-icall-cross-dso.c\00
// CHECK-DIAG: @[[TYPE:.*]] = private unnamed_addr constant { i16, i16, [{{.*}} x i8] } { i16 -1, i16 0, [{{.*}} x i8] c"'void ()'\00"
// CHECK-DIAG: @[[DATA:.*]] = private unnamed_addr global {{.*}}@[[SRC]]{{.*}}@[[TYPE]]


// ITANIUM: call i1 @llvm.type.test(i8* %{{.*}}, metadata !"_ZTSFvE"), !nosanitize
// ITANIUM-DIAG: call void @__cfi_slowpath_diag(i64 6588678392271548388, i8* %{{.*}}, {{.*}}@[[DATA]]{{.*}}) {{.*}}, !nosanitize
// ITANIUM-TRAP: call void @__cfi_slowpath(i64 6588678392271548388, i8* %{{.*}}) {{.*}}, !nosanitize

// MS: call i1 @llvm.type.test(i8* %{{.*}}, metadata !"?6AX@Z"), !nosanitize
// MS-DIAG: call void @__cfi_slowpath_diag(i64 4195979634929632483, i8* %{{.*}}, {{.*}}@[[DATA]]{{.*}}) {{.*}}, !nosanitize
// MS-TRAP: call void @__cfi_slowpath(i64 4195979634929632483, i8* %{{.*}}) {{.*}}, !nosanitize

void caller(void (*f)()) {
  f();
}

// Check that we emit both string and hash based type entries for static void g(),
// and don't emit them for the declaration of h().

// CHECK: define internal void @g({{.*}} !type [[TVOID:![0-9]+]] !type [[TVOID_GENERALIZED:![0-9]+]] !type [[TVOID_ID:![0-9]+]]
static void g(void) {}

// CHECK: declare {{(dso_local )?}}void @h({{[^!]*$}}
void h(void);

typedef void (*Fn)(void);
Fn g1() {
  return &g;
}
Fn h1() {
  return &h;
}

// CHECK: define {{(dso_local )?}}void @bar({{.*}} !type [[TNOPROTO:![0-9]+]] !type [[TNOPROTO_GENERALIZED:![0-9]+]] !type [[TNOPROTO_ID:![0-9]+]]
// ITANIUM: define available_externally void @foo({{[^!]*$}}
// MS: define linkonce_odr dso_local void @foo({{.*}} !type [[TNOPROTO]] !type [[TNOPROTO_GENERALIZED:![0-9]+]] !type [[TNOPROTO_ID]]
inline void foo() {}
void bar() { foo(); }

// CHECK: !{i32 4, !"Cross-DSO CFI", i32 1}

// Check that the type entries are correct.

// ITANIUM: [[TVOID]] = !{i64 0, !"_ZTSFvvE"}
// ITANIUM: [[TVOID_GENERALIZED]] = !{i64 0, !"_ZTSFvvE.generalized"}
// ITANIUM: [[TVOID_ID]] = !{i64 0, i64 9080559750644022485}
// ITANIUM: [[TNOPROTO]] = !{i64 0, !"_ZTSFvE"}
// ITANIUM: [[TNOPROTO_GENERALIZED]] = !{i64 0, !"_ZTSFvE.generalized"}
// ITANIUM: [[TNOPROTO_ID]] = !{i64 0, i64 6588678392271548388}

// MS: [[TVOID]] = !{i64 0, !"?6AXXZ"}
// MS: [[TVOID_GENERALIZED]] = !{i64 0, !"?6AXXZ.generalized"}
// MS: [[TVOID_ID]] = !{i64 0, i64 5113650790573562461}
// MS: [[TNOPROTO]] = !{i64 0, !"?6AX@Z"}
// MS: [[TNOPROTO_GENERALIZED]] = !{i64 0, !"?6AX@Z.generalized"}
// MS: [[TNOPROTO_ID]] = !{i64 0, i64 4195979634929632483}
