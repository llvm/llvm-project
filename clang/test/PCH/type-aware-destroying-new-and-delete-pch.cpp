// Test this without pch.
// RUN: %clang_cc1 -x c++ -std=c++26 -include %S/Inputs/type_aware_destroying_new_delete.h -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -std=c++26 -emit-pch -o %t %S/Inputs/type_aware_destroying_new_delete.h
// RUN: %clang_cc1 -x c++ -std=c++26 -include-pch %t -emit-llvm -o - %s 

// RUN: %clang_cc1 -x c++ -std=c++11 -emit-pch -fpch-instantiate-templates -o %t %S/Inputs/type_aware_destroying_new_delete.h
// RUN: %clang_cc1 -x c++ -std=c++11 -include-pch %t -emit-llvm -o - %s


static void call_in_pch_function(void) {
    in_pch_tests();
}

void out_of_pch_tests() {
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
