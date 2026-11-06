// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple i386-linux-gnu -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c++ -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows-pc -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS

// WINDOWS-NOT: "fmv-features"
// WINDOWS-NOT: "fmv-priority"

#ifdef __cplusplus
extern "C" {
#endif

// Haswell and v3 have equal runtime priority. Their source order breaks ties.
// A declaration followed by a definition must retain the same metadata.
__attribute__((target_clones("default,arch=haswell,arch=x86-64-v3")))
int haswell_first(int);
int haswell_first(int x) { return x; }

__attribute__((target_clones("default,arch=x86-64-v3,arch=haswell")))
int v3_first(int x) { return x; }

#ifdef __cplusplus
}
#endif

// CHECK: define{{.*}} i32 @haswell_first.default.2({{.*}}) #[[DEFAULT:[0-9]+]]
// CHECK: define{{.*}} i32 @haswell_first.arch_haswell.0({{.*}}) #[[H_FIRST:[0-9]+]]
// CHECK: define{{.*}} i32 @haswell_first.arch_x86-64-v3.1({{.*}}) #[[V_SECOND:[0-9]+]]
// CHECK: define{{.*}} i32 @v3_first.default.2({{.*}}) #[[DEFAULT]]
// CHECK: define{{.*}} i32 @v3_first.arch_x86-64-v3.0({{.*}}) #[[V_FIRST:[0-9]+]]
// CHECK: define{{.*}} i32 @v3_first.arch_haswell.1({{.*}}) #[[H_SECOND:[0-9]+]]
// CHECK: attributes #[[DEFAULT]] = { {{.*}}"fmv-features" "fmv-priority"="4294967295"
// CHECK: attributes #[[H_FIRST]] = { {{.*}}"fmv-features"="arch=haswell" "fmv-priority"="[[#PRIORITY:]]"
// CHECK: attributes #[[V_SECOND]] = { {{.*}}"fmv-features"="x86-64-v3" "fmv-priority"="[[#PRIORITY - 1]]"
// CHECK: attributes #[[V_FIRST]] = { {{.*}}"fmv-features"="x86-64-v3" "fmv-priority"="[[#PRIORITY]]"
// CHECK: attributes #[[H_SECOND]] = { {{.*}}"fmv-features"="arch=haswell" "fmv-priority"="[[#PRIORITY - 1]]"
