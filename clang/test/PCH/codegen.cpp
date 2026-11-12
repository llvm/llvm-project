// This test is the PCH version of Modules/codegen-flags.test . It uses its inputs.
// The purpose of this test is just verify that the codegen options work with PCH as well.
// All the codegen functionality should be tested in Modules/.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fmodules-codegen -x c++-header -building-pch-with-obj -emit-pch %S/../Modules/Inputs/codegen-flags/foo.h -o %t/foo-cg.pch
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fmodules-debuginfo -x c++-header -building-pch-with-obj -emit-pch %S/../Modules/Inputs/codegen-flags/foo.h -o %t/foo-di.pch

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=limited -o - -x precompiled-header %t/foo-cg.pch | FileCheck --check-prefix=CG %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=limited -o - -x precompiled-header %t/foo-di.pch | FileCheck --check-prefix=DI %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=limited -o - -include-pch %t/foo-cg.pch %S/../Modules/Inputs/codegen-flags/use.cpp | FileCheck --check-prefix=CG-USE %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=limited -o - -include-pch %t/foo-di.pch %S/../Modules/Inputs/codegen-flags/use.cpp | FileCheck --check-prefix=DI-USE %s

// Test with template instantiation in the pch.

// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fmodules-codegen -x c++-header -building-pch-with-obj -emit-pch -fpch-instantiate-templates %S/../Modules/Inputs/codegen-flags/foo.h -o %t/foo-cg.pch
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -fmodules-debuginfo -x c++-header -building-pch-with-obj -emit-pch -fpch-instantiate-templates %S/../Modules/Inputs/codegen-flags/foo.h -o %t/foo-di.pch

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=limited -o - -x precompiled-header %t/foo-cg.pch | FileCheck --check-prefix=CG %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=limited -o - -x precompiled-header %t/foo-di.pch | FileCheck --check-prefix=DI %s

// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=limited -o - -include-pch %t/foo-cg.pch %S/../Modules/Inputs/codegen-flags/use.cpp | FileCheck --check-prefix=CG-USE %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=limited -o - -include-pch %t/foo-di.pch %S/../Modules/Inputs/codegen-flags/use.cpp | FileCheck --check-prefix=DI-USE %s


// CG: define weak_odr void @_Z2f1v
// CG: DICompileUnit
// CG-NOT: DICompositeType

// CG-USE: declare void @_Z2f1v
// CG-USE: DICompileUnit
// CG-USE: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo"

// DI-NOT: define
// DI: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo"

// DI-USE: define linkonce_odr void @_Z2f1v
// DI-USE: = !DICompositeType(tag: DW_TAG_structure_type, name: "foo", {{.*}}, flags: DIFlagFwdDecl
