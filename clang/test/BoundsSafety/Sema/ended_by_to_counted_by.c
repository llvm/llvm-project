
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

// expected-no-diagnostics

#include <ptrcheck.h>

struct CountedData {
    int *__counted_by(len) buf;
    int len;
};

struct EndedData {
    int *__ended_by(end) start;
    int *end;
};

void Foo(struct CountedData *cp, struct EndedData *ep) {
    cp->buf = ep->start;
    cp->len = 10;
}

int glen;
void Bar(struct CountedData *cp, struct EndedData *ep) {
    cp->buf = ep->end;
    cp->len = glen;
}
