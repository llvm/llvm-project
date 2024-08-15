// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++20 -verify=ref,both %s




/// FIXME: The new interpreter also emits the "address of weak declaration" note in the pointer-to-bool case.

[[gnu::weak]] extern int a;
int ha[(bool)&a]; // both-warning {{variable length arrays in C++ are a Clang extension}} \
                  // expected-note {{comparison against address of weak declaration}} \
                  // both-error {{variable length array declaration not allowed at file scope}}
int ha2[&a == nullptr]; // both-warning {{variable length arrays in C++ are a Clang extension}} \
                        // both-note {{comparison against address of weak declaration '&a' can only be performed at runtime}} \
                        // both-error {{variable length array declaration not allowed at file scope}}
