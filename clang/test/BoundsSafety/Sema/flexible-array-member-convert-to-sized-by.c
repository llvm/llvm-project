

// RUN: %clang_cc1 -ast-dump -fbounds-safety -verify %s

// expected-no-diagnostics

#include <ptrcheck.h>
#include <stdint.h>

struct flex {
	uint8_t count;
	uint8_t body[__counted_by(count - 1)];
};

void pass_ptr(const void *__sized_by(size), unsigned size);

void foo(struct flex *f) {
	pass_ptr(f, 8);
}
