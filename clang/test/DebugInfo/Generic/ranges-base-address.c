// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - | FileCheck --check-prefix=NORNGBSE %s
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o - -fdebug-ranges-base-address | FileCheck --check-prefix=RNGBSE %s

// NORNGBSE-NOT: rangesBaseAddress
// RNGBSE: !DICompileUnit({{.*}}, rangesBaseAddress: true

void f1(void) {
}

