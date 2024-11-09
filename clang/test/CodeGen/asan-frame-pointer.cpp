/// -mframe-pointer=none sets the module flag "frame-pointer" (merge behavior: max).
/// asan synthesized ctor/dtor get the "frame-pointer" function attribute if not zero (default).
// RUN: %clang_cc1 -emit-llvm -fsanitize=address -mframe-pointer=none %s -o - | FileCheck %s --check-prefix=NONE
// RUN: %clang_cc1 -emit-llvm -fsanitize=address -mframe-pointer=non-leaf %s -o - | FileCheck %s --check-prefix=NONLEAF
// RUN: %clang_cc1 -emit-llvm -fsanitize=address -mframe-pointer=all %s -o - | FileCheck %s --check-prefix=ALL

int global;

// NONE: define internal void @asan.module_ctor() #[[#ATTR:]] {
// NONE: define internal void @asan.module_dtor() #[[#ATTR]] {
// NONE: attributes #[[#ATTR]] = { nounwind

// NONLEAF: define internal void @asan.module_ctor() #[[#ATTR:]] {
// NONLEAF: define internal void @asan.module_dtor() #[[#ATTR]] {
// NONLEAF: attributes #[[#ATTR]] = { nounwind "frame-pointer"="non-leaf"

// ALL: define internal void @asan.module_ctor() #[[#ATTR:]] {
// ALL: define internal void @asan.module_dtor() #[[#ATTR]] {
// ALL: attributes #[[#ATTR]] = { nounwind "frame-pointer"="all"
