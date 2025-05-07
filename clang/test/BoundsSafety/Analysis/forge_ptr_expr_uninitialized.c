

// RUN: %clang_cc1 -analyze -analyzer-checker=core.uninitialized.UndefReturn -fbounds-safety -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core.uninitialized.UndefReturn -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// XFAIL: *
// rdar://72163355
int foo() {
    int arr[10];
    int a = 10;
    int *ptr = __builtin_unsafe_forge_bidi_indexable(arr, 4 * sizeof(int));
    // expected-warning@+1{{Undefined or garbage value returned to caller [core.uninitialized.UndefReturn]}}
    return arr[a];
}
