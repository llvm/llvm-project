// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature -fmv -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-NOFMV

int implicit_default_decl_first(void);
__attribute__((target_version("default"))) int implicit_default_decl_first(void) { return 1; }
int caller1(void) { return implicit_default_decl_first(); }

__attribute__((target_version("default"))) int explicit_default_def_first(void) { return 2; }
int explicit_default_def_first(void);
int caller2(void) { return explicit_default_def_first(); }

int implicit_default_def_first(void) { return 3; }
__attribute__((target_version("default"))) int implicit_default_def_first(void);
int caller3(void) { return implicit_default_def_first(); }

__attribute__((target_version("default"))) int explicit_default_decl_first(void);
int explicit_default_decl_first(void) { return 4; }
int caller4(void) { return explicit_default_decl_first(); }

int no_def_implicit_default_first(void);
__attribute__((target_version("default"))) int no_def_implicit_default_first(void);
int caller5(void) { return no_def_implicit_default_first(); }

__attribute__((target_version("default"))) int no_def_explicit_default_first(void);
int no_def_explicit_default_first(void);
int caller6(void) { return no_def_explicit_default_first(); }
//.
// CHECK: @implicit_default_decl_first = weak_odr ifunc i32 (), ptr @implicit_default_decl_first.resolver
// CHECK: @explicit_default_def_first = weak_odr ifunc i32 (), ptr @explicit_default_def_first.resolver
// CHECK: @implicit_default_def_first = weak_odr ifunc i32 (), ptr @implicit_default_def_first.resolver
// CHECK: @explicit_default_decl_first = weak_odr ifunc i32 (), ptr @explicit_default_decl_first.resolver
//.
// CHECK: Function Attrs: noinline nounwind optnone
// CHECK-LABEL: define {{[^@]+}}@implicit_default_decl_first.default
// CHECK-SAME: () #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 1
//
//
// CHECK: Function Attrs: noinline nounwind optnone
// CHECK-LABEL: define {{[^@]+}}@caller1
// CHECK-SAME: () #[[ATTR1:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CALL:%.*]] = call i32 @implicit_default_decl_first()
// CHECK-NEXT:    ret i32 [[CALL]]
//
//
// CHECK: Function Attrs: noinline nounwind optnone
// CHECK-LABEL: define {{[^@]+}}@explicit_default_def_first.default
// CHECK-SAME: () #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 2
//
//
// CHECK: Function Attrs: noinline nounwind optnone
// CHECK-LABEL: define {{[^@]+}}@caller2
// CHECK-SAME: () #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CALL:%.*]] = call i32 @explicit_default_def_first()
// CHECK-NEXT:    ret i32 [[CALL]]
//
//
// CHECK: Function Attrs: noinline nounwind optnone
// CHECK-LABEL: define {{[^@]+}}@implicit_default_def_first.default
// CHECK-SAME: () #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 3
//
//
// CHECK: Function Attrs: noinline nounwind optnone
// CHECK-LABEL: define {{[^@]+}}@caller3
// CHECK-SAME: () #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CALL:%.*]] = call i32 @implicit_default_def_first()
// CHECK-NEXT:    ret i32 [[CALL]]
//
//
// CHECK: Function Attrs: noinline nounwind optnone
// CHECK-LABEL: define {{[^@]+}}@explicit_default_decl_first.default
// CHECK-SAME: () #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    ret i32 4
//
//
// CHECK: Function Attrs: noinline nounwind optnone
// CHECK-LABEL: define {{[^@]+}}@caller4
// CHECK-SAME: () #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CALL:%.*]] = call i32 @explicit_default_decl_first()
// CHECK-NEXT:    ret i32 [[CALL]]
//
//
// CHECK: declare i32 @no_def_implicit_default_first() #[[ATTR2:[0-9]+]]
//
//
// CHECK: Function Attrs: noinline nounwind optnone
// CHECK-LABEL: define {{[^@]+}}@caller5
// CHECK-SAME: () #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CALL:%.*]] = call i32 @no_def_implicit_default_first()
// CHECK-NEXT:    ret i32 [[CALL]]
//
//
// CHECK: declare i32 @no_def_explicit_default_first() #[[ATTR2]]
//
//
// CHECK: Function Attrs: noinline nounwind optnone
// CHECK-LABEL: define {{[^@]+}}@caller6
// CHECK-SAME: () #[[ATTR1]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[CALL:%.*]] = call i32 @no_def_explicit_default_first()
// CHECK-NEXT:    ret i32 [[CALL]]
//
//
// CHECK-LABEL: define {{[^@]+}}@implicit_default_decl_first.resolver() comdat {
// CHECK-NEXT:  resolver_entry:
// CHECK-NEXT:    ret ptr @implicit_default_decl_first.default
//
//
// CHECK-LABEL: define {{[^@]+}}@explicit_default_def_first.resolver() comdat {
// CHECK-NEXT:  resolver_entry:
// CHECK-NEXT:    ret ptr @explicit_default_def_first.default
//
//
// CHECK-LABEL: define {{[^@]+}}@implicit_default_def_first.resolver() comdat {
// CHECK-NEXT:  resolver_entry:
// CHECK-NEXT:    ret ptr @implicit_default_def_first.default
//
//
// CHECK-LABEL: define {{[^@]+}}@explicit_default_decl_first.resolver() comdat {
// CHECK-NEXT:  resolver_entry:
// CHECK-NEXT:    ret ptr @explicit_default_decl_first.default
//
//
// CHECK: declare i32 @no_def_implicit_default_first.default() #[[ATTR2]]
//
//
// CHECK: declare i32 @no_def_explicit_default_first.default() #[[ATTR2]]
//
//
// CHECK-NOFMV: Function Attrs: noinline nounwind optnone
// CHECK-NOFMV-LABEL: define {{[^@]+}}@caller1
// CHECK-NOFMV-SAME: () #[[ATTR0:[0-9]+]] {
// CHECK-NOFMV-NEXT:  entry:
// CHECK-NOFMV-NEXT:    [[CALL:%.*]] = call i32 @implicit_default_decl_first()
// CHECK-NOFMV-NEXT:    ret i32 [[CALL]]
//
//
// CHECK-NOFMV: Function Attrs: noinline nounwind optnone
// CHECK-NOFMV-LABEL: define {{[^@]+}}@implicit_default_decl_first
// CHECK-NOFMV-SAME: () #[[ATTR1:[0-9]+]] {
// CHECK-NOFMV-NEXT:  entry:
// CHECK-NOFMV-NEXT:    ret i32 1
//
//
// CHECK-NOFMV: Function Attrs: noinline nounwind optnone
// CHECK-NOFMV-LABEL: define {{[^@]+}}@caller2
// CHECK-NOFMV-SAME: () #[[ATTR0]] {
// CHECK-NOFMV-NEXT:  entry:
// CHECK-NOFMV-NEXT:    [[CALL:%.*]] = call i32 @explicit_default_def_first()
// CHECK-NOFMV-NEXT:    ret i32 [[CALL]]
//
//
// CHECK-NOFMV: Function Attrs: noinline nounwind optnone
// CHECK-NOFMV-LABEL: define {{[^@]+}}@explicit_default_def_first
// CHECK-NOFMV-SAME: () #[[ATTR1]] {
// CHECK-NOFMV-NEXT:  entry:
// CHECK-NOFMV-NEXT:    ret i32 2
//
//
// CHECK-NOFMV: Function Attrs: noinline nounwind optnone
// CHECK-NOFMV-LABEL: define {{[^@]+}}@implicit_default_def_first
// CHECK-NOFMV-SAME: () #[[ATTR0]] {
// CHECK-NOFMV-NEXT:  entry:
// CHECK-NOFMV-NEXT:    ret i32 3
//
//
// CHECK-NOFMV: Function Attrs: noinline nounwind optnone
// CHECK-NOFMV-LABEL: define {{[^@]+}}@caller3
// CHECK-NOFMV-SAME: () #[[ATTR0]] {
// CHECK-NOFMV-NEXT:  entry:
// CHECK-NOFMV-NEXT:    [[CALL:%.*]] = call i32 @implicit_default_def_first()
// CHECK-NOFMV-NEXT:    ret i32 [[CALL]]
//
//
// CHECK-NOFMV: Function Attrs: noinline nounwind optnone
// CHECK-NOFMV-LABEL: define {{[^@]+}}@caller4
// CHECK-NOFMV-SAME: () #[[ATTR0]] {
// CHECK-NOFMV-NEXT:  entry:
// CHECK-NOFMV-NEXT:    [[CALL:%.*]] = call i32 @explicit_default_decl_first()
// CHECK-NOFMV-NEXT:    ret i32 [[CALL]]
//
//
// CHECK-NOFMV: Function Attrs: noinline nounwind optnone
// CHECK-NOFMV-LABEL: define {{[^@]+}}@explicit_default_decl_first
// CHECK-NOFMV-SAME: () #[[ATTR1]] {
// CHECK-NOFMV-NEXT:  entry:
// CHECK-NOFMV-NEXT:    ret i32 4
//
//
// CHECK-NOFMV: Function Attrs: noinline nounwind optnone
// CHECK-NOFMV-LABEL: define {{[^@]+}}@caller5
// CHECK-NOFMV-SAME: () #[[ATTR0]] {
// CHECK-NOFMV-NEXT:  entry:
// CHECK-NOFMV-NEXT:    [[CALL:%.*]] = call i32 @no_def_implicit_default_first()
// CHECK-NOFMV-NEXT:    ret i32 [[CALL]]
//
//
// CHECK-NOFMV: declare i32 @no_def_implicit_default_first() #[[ATTR2:[0-9]+]]
//
//
// CHECK-NOFMV: Function Attrs: noinline nounwind optnone
// CHECK-NOFMV-LABEL: define {{[^@]+}}@caller6
// CHECK-NOFMV-SAME: () #[[ATTR0]] {
// CHECK-NOFMV-NEXT:  entry:
// CHECK-NOFMV-NEXT:    [[CALL:%.*]] = call i32 @no_def_explicit_default_first()
// CHECK-NOFMV-NEXT:    ret i32 [[CALL]]
//
//
// CHECK-NOFMV: declare i32 @no_def_explicit_default_first() #[[ATTR2]]
//.
