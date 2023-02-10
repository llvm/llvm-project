// RUN: %clang -O -fexperimental-sanitize-metadata=all -target x86_64-gnu-linux -x c -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALLOW
// RUN: echo "fun:foo" > %t.fun
// RUN: %clang -O -fexperimental-sanitize-metadata=all -fexperimental-sanitize-metadata-ignorelist=%t.fun -target x86_64-gnu-linux -x c -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=FUN
// RUN: echo "src:*sanitize-metadata-ignorelist.c" > %t.src
// RUN: %clang -O -fexperimental-sanitize-metadata=all -fexperimental-sanitize-metadata-ignorelist=%t.src -target x86_64-gnu-linux -x c -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=SRC

int y;

// ALLOW-LABEL: define {{[^@]+}}@foo
// ALLOW-SAME: () local_unnamed_addr #[[ATTR0:[0-9]+]] !pcsections !5 {
// ALLOW-NEXT:  entry:
// ALLOW-NEXT:    [[TMP0:%.*]] = atomicrmw add ptr @y, i32 1 monotonic, align 4, !pcsections !7
// ALLOW-NEXT:    ret void
//
// FUN-LABEL: define {{[^@]+}}@foo
// FUN-SAME: () local_unnamed_addr #[[ATTR0:[0-9]+]] {
// FUN-NEXT:  entry:
// FUN-NEXT:    [[TMP0:%.*]] = atomicrmw add ptr @y, i32 1 monotonic, align 4
// FUN-NEXT:    ret void
//
// SRC-LABEL: define {{[^@]+}}@foo
// SRC-SAME: () local_unnamed_addr #[[ATTR0:[0-9]+]] {
// SRC-NEXT:  entry:
// SRC-NEXT:    [[TMP0:%.*]] = atomicrmw add ptr @y, i32 1 monotonic, align 4
// SRC-NEXT:    ret void
//
void foo() {
  __atomic_fetch_add(&y, 1, __ATOMIC_RELAXED);
}

// ALLOW-LABEL: define {{[^@]+}}@bar
// ALLOW-SAME: () local_unnamed_addr #[[ATTR0]] !pcsections !5 {
// ALLOW-NEXT:  entry:
// ALLOW-NEXT:    [[TMP0:%.*]] = atomicrmw add ptr @y, i32 2 monotonic, align 4, !pcsections !7
// ALLOW-NEXT:    ret void
//
// FUN-LABEL: define {{[^@]+}}@bar
// FUN-SAME: () local_unnamed_addr #[[ATTR0]] !pcsections !5 {
// FUN-NEXT:  entry:
// FUN-NEXT:    [[TMP0:%.*]] = atomicrmw add ptr @y, i32 2 monotonic, align 4, !pcsections !7
// FUN-NEXT:    ret void
//
// SRC-LABEL: define {{[^@]+}}@bar
// SRC-SAME: () local_unnamed_addr #[[ATTR0]] {
// SRC-NEXT:  entry:
// SRC-NEXT:    [[TMP0:%.*]] = atomicrmw add ptr @y, i32 2 monotonic, align 4
// SRC-NEXT:    ret void
//
void bar() {
  __atomic_fetch_add(&y, 2, __ATOMIC_RELAXED);
}

// ALLOW: __sanitizer_metadata_covered.module_ctor
// FUN: __sanitizer_metadata_covered.module_ctor
// SRC-NOT: __sanitizer_metadata_covered.module_ctor
