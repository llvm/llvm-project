

// RUN: %clang_cc1 -analyze -analyzer-checker=deadcode.DeadStores -fbounds-safety -verify %s
// RUN: %clang_cc1 -analyze -analyzer-checker=deadcode.DeadStores -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
int foo() {
    int arr[10];
    int a = 10;
    // expected-warning@+1{{Value stored to 'ptr' during its initialization is never read [deadcode.DeadStores]}}
    int *ptr = __builtin_unsafe_forge_bidi_indexable(arr, 4 * sizeof(int));
    return 0;
}
