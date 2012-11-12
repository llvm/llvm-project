// RUN: %clang_cc1 -fsyntax-only -verify %s
void* a(unsigned x) {
return __builtin_return_address(0);
}

void b(unsigned x) {
return __builtin_return_address(x); // expected-error{{argument to '__builtin_return_address' must be a constant integer}}
}

void* c(unsigned x) {
return __builtin_frame_address(0);
}

void d(unsigned x) {
return __builtin_frame_address(x); // expected-error{{argument to '__builtin_frame_address' must be a constant integer}}
}
