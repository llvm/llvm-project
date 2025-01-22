
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s

#include <stdint.h>
#include <ptrcheck.h>

#ifndef __bidi_indexable
#define __bidi_indexable
#endif

typedef struct {
    uint8_t dummy;
    uint32_t value;
} __attribute__((packed)) packed_t;

void external(void * __sized_by(size), int size);

void test(void) {
    packed_t s;
    const int size = 4;
    external(&s.value, size);
    void *local = &s.value;
}

void test_explicit_cast_single(void) {
    packed_t s;
    const int size = 4;
    external((uint32_t * __single) &s.value, size); // expected-warning{{taking address of packed member 'value' of class or structure 'packed_t' may result in an unaligned pointer value}}
    void *local = (uint32_t * __bidi_indexable)(uint32_t * __single) &s.value; // expected-warning{{taking address of packed member 'value' of class or structure 'packed_t' may result in an unaligned pointer value}}
}

void test_explicit_cast_bidi(void) {
    packed_t s;
    const int size = 4;
    external((uint32_t * __bidi_indexable) &s.value, size); // expected-warning{{taking address of packed member 'value' of class or structure 'packed_t' may result in an unaligned pointer value}}
    void *local = (uint32_t * __bidi_indexable) &s.value; // expected-warning{{taking address of packed member 'value' of class or structure 'packed_t' may result in an unaligned pointer value}}
}

void test_explicit_cast_sized(void) {
    packed_t s;
    const int size = 4;
    external((uint32_t * __sized_by(size)) &s.value, size); // expected-warning{{taking address of packed member 'value' of class or structure 'packed_t' may result in an unaligned pointer value}}
    void *local = (uint32_t * __sized_by(size)) &s.value; // expected-warning{{taking address of packed member 'value' of class or structure 'packed_t' may result in an unaligned pointer value}}
    external((void * __sized_by(size)) &s.value, size);
    local = (void * __sized_by(size)) &s.value;
}
