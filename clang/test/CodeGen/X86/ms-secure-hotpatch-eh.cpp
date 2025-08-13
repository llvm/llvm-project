// REQUIRES: x86-registered-target

// Global constant data such as exception handler tables should not be redirected by Windows Secure Hot-Patching
//
// RUN: %clang_cl -c --target=x86_64-windows-msvc /EHsc -O2 -fms-secure-hotpatch-functions-list=this_gets_hotpatched /Fo%t.obj /clang:-S /clang:-o- -- %s 2>& 1 | FileCheck %s

class Foo {
public:
    int x;
};

void this_might_throw();

extern "C" int this_gets_hotpatched(int k) {
    int ret;
    try {
        this_might_throw();
        ret = 1;
    } catch (Foo& f) {
        ret = 2;
    }
    return ret;
}

// We expect that RTTI data is not redirected.
// CHECK-NOT: "__ref_??_R0?AVFoo@@@8"
