// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-linux -emit-llvm %s \
// RUN:   -Wno-source-uses-openmp -o - | FileCheck %s

// Device and target_device ISA selectors are matched independently.

int variant(void);

// MATCH_ANY is satisfied by the supported device ISA.
#pragma omp declare variant(variant)                                           \
    match(device = {isa(sse2)}, target_device = {isa(bogus_isa)},              \
          implementation = {extension(match_any)})
int any_base(void);

// CHECK-LABEL: define{{.*}} i32 @any_caller(
// CHECK: call i32 @variant()
int any_caller(void) { return any_base(); }

// MATCH_NONE is not satisfied because the target_device ISA is supported.
#pragma omp declare variant(variant)                                           \
    match(device = {isa(bogus_isa)}, target_device = {isa(sse2)},              \
          implementation = {extension(match_none)})
int none_base(void);

// CHECK-LABEL: define{{.*}} i32 @none_caller(
// CHECK: call i32 @none_base()
int none_caller(void) { return none_base(); }

// BEGIN DECLARE VARIANT ignores target_device traits, so the unsupported
// target_device ISA does not elide the block.
int mixed(void) { return 0; }
#pragma omp begin declare variant                                              \
    match(device = {isa(sse2)}, target_device = {isa(bogus_isa)})
// CHECK: define{{.*}} i32 @"mixed$ompvariant${{.*}}"()
int mixed(void) { return 1; }
#pragma omp end declare variant
