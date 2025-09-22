// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -std=c++26 -fmodules-cache-path=%t -I %S/Inputs/PR137102 -emit-llvm-only %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -std=c++26 -fmodules-cache-path=%t -I %S/Inputs/PR137102 -emit-llvm-only %s -triple i686-windows

#include "type_aware_destroying_new_delete.h"


static void call_in_module_function(void) {
    in_module_tests();
}

void out_of_module_tests() {
    A* a = new A;
    delete a;
    B *b = new B;
    delete b;
    C *c = new C;
    delete c;
    D *d = new D;
    delete d;
    E *e = new E;
    delete e;
}
