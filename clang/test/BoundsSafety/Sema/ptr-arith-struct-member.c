
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

struct Foo { int *buf; int cnt; };

void Test(struct Foo *f) {
    // expected-error@+2{{pointer arithmetic on single pointer 'f->buf' is out of bounds; consider adding '__counted_by' to 'Foo::buf'}}
    // expected-note@-4{{pointer 'Foo::buf' declared here}}
    int *ptr = f->buf + 2;
}
