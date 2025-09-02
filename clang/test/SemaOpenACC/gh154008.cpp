// RUN: %clang_cc1 %s -fopenacc -verify

// expected-error@+2{{expected ';'}}
// expected-error@+1{{blocks support disabled}}
void *a = ^ { static int b };
