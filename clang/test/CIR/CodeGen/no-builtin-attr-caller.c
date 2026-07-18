// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fno-builtin-memcpy %s -o %t.mc.cir
// RUN: FileCheck --input-file=%t.mc.cir %s --check-prefix=MERGE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fno-builtin %s -o %t.all.cir
// RUN: FileCheck --input-file=%t.all.cir %s --check-prefix=ALL

// A caller's no_builtin list rides on the calls it makes. A -fno-builtin-<name>
// list on the call merges with it, and any empty list disables everything.

unsigned long strlen(const char *);

// CHECK-LABEL: @named
// CHECK: cir.call @strlen(%{{.*}}){{.*}}nobuiltins = ["strlen"]
// MERGE-LABEL: @named
// MERGE: cir.call @strlen(%{{.*}}){{.*}}nobuiltins = ["memcpy", "strlen"]
// ALL-LABEL: @named
// ALL: cir.call @strlen(%{{.*}}){{.*}}nobuiltins = []
__attribute__((no_builtin("strlen")))
unsigned long named(const char *s) { return strlen(s); }

// CHECK-LABEL: @all
// CHECK: cir.call @strlen(%{{.*}}){{.*}}nobuiltins = []
__attribute__((no_builtin))
unsigned long all(const char *s) { return strlen(s); }
