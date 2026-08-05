// Test C++ -gmodules debug info in the PCMs with local submodule visibility.
// UNSUPPORTED: target={{.*}}-aix{{.*}}
// REQUIRES: asserts
// RUN: rm -rf %t
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++14 \
// RUN:   -fmodules-local-submodule-visibility %s \
// RUN:   -dwarf-ext-refs -fmodule-format=obj -debug-info-kind=standalone \
// RUN:   -dwarf-version=4 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path="%t" -o %t.ll -I%S/Inputs/lsv-debuginfo \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-mod.ll
// RUN: cat %t-mod.ll | FileCheck %s

// RUN: rm -rf %t
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++14 \
// RUN:   -fmodules-local-submodule-visibility %s \
// RUN:   -dwarf-ext-refs -fmodule-format=obj -debug-info-kind=standalone \
// RUN:   -dwarf-version=4 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path="%t" -o %t.ll -I%S/Inputs/lsv-debuginfo \
// RUN:   -DWITH_NAMESPACE \
// RUN:   -mllvm -debug-only=pchcontainer &>%t-mod.ll
// RUN: cat %t-mod.ll | FileCheck %s

// ADT
// CHECK: !llvm.raw.sections = !{[[ADT_SEC:![0-9]+]]}
// CHECK: [[ADT_SEC]] = !{!"{{(__CLANG,)?(__)?}}clangast",

// B
// CHECK: !llvm.raw.sections = !{[[B_SEC:![0-9]+]]}

// This type isn't anchored anywhere, expect a full definition.
// CHECK: !DICompositeType({{.*}}, name: "AlignedCharArray<4U, 16U>",
// CHECK-SAME:             elements:
// CHECK: [[B_SEC]] = !{!"{{(__CLANG,)?(__)?}}clangast",

// C
// CHECK: !llvm.raw.sections = !{[[C_SEC:![0-9]+]]}

// Here, too.
// CHECK: !DICompositeType({{.*}}, name: "AlignedCharArray<4U, 16U>",
// CHECK-SAME:             elements:
// CHECK: [[C_SEC]] = !{!"{{(__CLANG,)?(__)?}}clangast",

#include <B/B.h>
