// This verifies different patterns of accesses to global variables within functions that are hotpatched
//
// RUN: %clang_cl -c --target=x86_64-windows-msvc -O2 -fms-hotpatch-functions-list=this_gets_hotpatched /Fo%t.obj /clang:-S /clang:-o- %s 2>& 1 | FileCheck %s

extern int g_foo;
extern int g_bar;

int* this_gets_hotpatched(int k, void g()) {
    g_foo = 10;

    int* ret;
    if (k) {
        g();
        ret = &g_foo;
    } else {
        ret = &g_bar;
    }
    return ret;
}
