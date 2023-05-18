// RUN: %clang_cc1 -fsyntax-only -verify %s

void* a(unsigned x) {
return __builtin_return_address(0);
}

void b(unsigned x) {
return __builtin_return_address(x); // expected-error{{argument to '__builtin_return_address' must be a constant integer}}
}

void* c(unsigned x) {
// expected-error@+1 {{argument value 4294967295 is outside the valid range [0, 65535]}}
return __builtin_return_address(-1);
}

void* d(unsigned x) {
// expected-error@+1 {{argument value 1048575 is outside the valid range [0, 65535]}}
return __builtin_return_address(0xFFFFF);
}

void* e(unsigned x) {
return __builtin_frame_address(0);
}

void f(unsigned x) {
// expected-error@+1 {{argument to '__builtin_frame_address' must be a constant integer}}
return __builtin_frame_address(x);
}

void* g(unsigned x) {
// expected-error@+1 {{argument value 4294967295 is outside the valid range [0, 65535]}}
return __builtin_frame_address(-1);
}

void* h(unsigned x) {
// expected-error@+1 {{argument value 1048575 is outside the valid range [0, 65535]}}
return __builtin_frame_address(0xFFFFF);
}
