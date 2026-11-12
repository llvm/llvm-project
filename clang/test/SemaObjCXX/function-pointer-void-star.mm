// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

extern "C" id (*_dealloc)(id) ;

void foo() {
        extern void *_original_dealloc;
        if (_dealloc == _original_dealloc) { }
        if (_dealloc != _original_dealloc) { }
}
