// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++20 -verify=ref,both %s

[[gnu::weak]] extern int a;
int ha[(bool)&a]; // both-warning {{variable length arrays in C++ are a Clang extension}} \
                  // both-error {{variable length array declaration not allowed at file scope}}
int ha2[&a == nullptr]; // both-warning {{variable length arrays in C++ are a Clang extension}} \
                        // both-note {{comparison against address of weak declaration '&a' can only be performed at runtime}} \
                        // both-error {{variable length array declaration not allowed at file scope}}

extern const int W1 __attribute__((weak)) = 10; // both-note {{declared here}}
static_assert(W1 == 10, ""); // both-error {{static assertion expression is not an integral constant expression}} \
                             // both-note {{initializer of weak variable 'W1' is not considered constant because it may be different at runtime}}

extern const int W2 __attribute__((weak)); // both-note {{declared here}}
static_assert(W2 == 10, ""); // both-error {{static assertion expression is not an integral constant expression}} \
                             // both-note {{initializer of 'W2' is unknown}}
