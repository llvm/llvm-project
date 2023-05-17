// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %clang_cc1 -x objective-c-header -emit-pch %S/Inputs/pch-used.h -o %t/pch-used.h.pch -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -O0 -isystem %S/Inputs/System/usr/include
// RUN: %clang_cc1 %s -include-pch %t/pch-used.h.pch -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -O0 -isystem %S/Inputs/System/usr/include -emit-llvm -o - | FileCheck %s

void f(void) { SPXTrace(); }
void g(void) { double x = DBL_MAX; }

// CHECK: define internal {{.*}}void @SPXTrace
