// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s -DEMPTY | FileCheck %s --check-prefix EMPTY
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s -DSYM_ONLY | FileCheck %s --check-prefix SYM_ONLY
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s -DSYMVER_ONLY | FileCheck %s --check-prefix SYMVER_ONLY
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix CHECK

// REQUIRES: aarch64-registered-target

#ifdef EMPTY
asm (
    ".text" "\n"
    ".previous" "\n"
);
// Inline assembly does not define any symbols. Flags are set, but empty.
//
// EMPTY: module asm ".text"
// EMPTY: module asm ".previous"
// EMPTY: !{{.*}} = !{i32 6, !"global-asm-symbols", ![[NONE:[0-9]+]]}
// EMPTY: ![[NONE]] = !{}
// EMPTY: !{{.*}} = !{i32 6, !"global-asm-symvers", ![[NONE]]}
#endif


#ifdef SYM_ONLY
asm (
    ".text" "\n"
    "foo:" "\n"
    ".previous" "\n"
);
// Inline assembly defines symbols, but not symvers. Flags are set, but symvers
// is empty.
//
// SYM_ONLY: module asm ".text"
// SYM_ONLY: module asm "foo:"
// SYM_ONLY: module asm ".previous"
// SYM_ONLY: !{{.*}} = !{i32 6, !"global-asm-symbols", ![[SYM:[0-9]+]]}
// SYM_ONLY: ![[SYM]] = !{![[FOO:[0-9]+]]}
// SYM_ONLY: ![[FOO]] = !{!"foo", i32 2048}
// SYM_ONLY: !{{.*}} = !{i32 6, !"global-asm-symvers", ![[NONE:[0-9]+]]}
// SYM_ONLY: ![[NONE]] = !{}
#endif


#ifdef SYMVER_ONLY
asm (
    ".text" "\n"
    ".symver foo, foo@VER" "\n"
    ".previous" "\n"
);
// Inline assembly defines symvers. The corresponding symbol is implicitly
// declared as "undefined global".
//
// SYMVER_ONLY: module asm ".text"
// SYMVER_ONLY: module asm ".symver foo, foo@VER"
// SYMVER_ONLY: module asm ".previous"
// SYMVER_ONLY: !{{.*}} = !{i32 6, !"global-asm-symbols", ![[SYM:[0-9]+]]}
// SYMVER_ONLY: ![[SYM]] = !{![[FOO_UNDEF:[0-9]+]]}
// SYMVER_ONLY: ![[FOO_UNDEF]] = !{!"foo", i32 2051}
// SYMVER_ONLY: !{{.*}} = !{i32 6, !"global-asm-symvers", ![[SYMVERS:[0-9]+]]}
// SYMVER_ONLY: ![[SYMVERS]] = !{![[FOO_VER:[0-9]+]]}
// SYMVER_ONLY: ![[FOO_VER]] = !{!"foo", !"foo@VER"}
#endif

// If there is no inline assembly, module flags should be omitted.
// CHECK-NOT: module asm
// CHECK-NOT: global-asm

void bar() {}
