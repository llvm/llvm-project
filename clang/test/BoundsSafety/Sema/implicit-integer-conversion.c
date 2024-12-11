

// RUN: %clang_cc1 -fbounds-safety -verify -Wshorten-64-to-32 -triple x86_64 -fsyntax-only %s

#include <ptrcheck.h>
#include <stdint.h>

void foo(int *__counted_by(size) buf, uint32_t size);

void bar(int *__sized_by(len) arr, uint64_t len) {
    foo(arr, len); // expected-warning{{implicit conversion loses integer precision: 'uint64_t' (aka 'unsigned long') to 'uint32_t' (aka 'unsigned int')}}
}
