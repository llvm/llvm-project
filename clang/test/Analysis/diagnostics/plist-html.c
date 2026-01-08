// RUN: rm -rf %t && mkdir %t && %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist-html %s -o %t/plist-html.c.plist -verify
// RUN: tail -n +11 %t/plist-html.c.plist | %normalize_plist | diff -ub %S/Inputs/expected-plists/plist-html.c.plist -

int foo(int *p) {
    if (p) {
        return 0;
    } else {
        return *p;  // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
    }
}
