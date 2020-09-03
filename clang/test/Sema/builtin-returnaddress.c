// RUN: %clang_cc1 -fsyntax-only -Wframe-address -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wmost -verify %s

void* a(unsigned x) {
return __builtin_return_address(0);
}

void* b(unsigned x) {
return __builtin_return_address(1); // expected-warning{{calling '__builtin_return_address' with a nonzero argument is unsafe}}
}

void* c(unsigned x) {
return __builtin_frame_address(0);
}

void* d(unsigned x) {
return __builtin_frame_address(1); // expected-warning{{calling '__builtin_frame_address' with a nonzero argument is unsafe}}
}

