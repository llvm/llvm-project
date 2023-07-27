// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -verify -fblocks -triple x86_64-apple-darwin10.0.0 %s
// rdar://10187884

typedef void (^blk)(id, __attribute((ns_consumed)) id);
typedef void (^blk1)(__attribute((ns_consumed))id, __attribute((ns_consumed)) id);
blk a = ^void (__attribute((ns_consumed)) id, __attribute((ns_consumed)) id){}; // expected-error {{cannot initialize a variable of type '__strong blk'}}

blk b = ^void (id, __attribute((ns_consumed)) id){};

blk c = ^void (__attribute((ns_consumed)) id, __attribute((ns_consumed)) id){}; // expected-error {{cannot initialize a variable of type '__strong blk'}}

blk d = ^void (id, id) {}; // expected-error {{cannot initialize a variable of type '__strong blk'}}

blk1 a1 = ^void (__attribute((ns_consumed)) id, id){}; // expected-error {{cannot initialize a variable of type '__strong blk1'}}

blk1 b2 = ^void (id, __attribute((ns_consumed)) id){}; // expected-error {{cannot initialize a variable of type '__strong blk1'}}

blk1 c3 = ^void (__attribute((ns_consumed)) id, __attribute((ns_consumed)) id){};

blk1 d4 = ^void (id, id) {}; // expected-error {{cannot initialize a variable of type '__strong blk1'}}


typedef void (*releaser_t)(__attribute__((ns_consumed)) id);

void normalFunction(id);
releaser_t r1 = normalFunction; // expected-error {{cannot initialize a variable of type 'releaser_t'}}

void releaser(__attribute__((ns_consumed)) id);
releaser_t r2 = releaser; // no-warning

template <typename T>
void templateFunction(T) { } // expected-note {{candidate template ignored: could not match 'void (__strong id)' against 'void (__attribute__((ns_consumed)) id)'}} \
                             // expected-note {{candidate template ignored: could not match 'void (AntiRelease *__strong)' against 'void (__attribute__((ns_consumed)) AntiRelease *__strong)'}}
releaser_t r3 = templateFunction<id>; // expected-error {{address of overloaded function 'templateFunction' does not match required type 'void (__attribute__((ns_consumed)) id)'}}

template <typename T>
void templateReleaser(__attribute__((ns_consumed)) T) { }
// expected-note@-1 {{candidate template ignored: could not match 'void (__attribute__((ns_consumed)) AntiRelease *__strong)' against 'void (AntiRelease *__strong)'}}
// expected-note@-2 {{candidate template ignored: could not match 'void (__attribute__((ns_consumed)) ExplicitAntiRelease *__strong)' against 'void (ExplicitAntiRelease *__strong)'}}
releaser_t r4 = templateReleaser<id>; // no-warning


@class AntiRelease, ExplicitAntiRelease, ProRelease;

template<>
void templateFunction(__attribute__((ns_consumed)) AntiRelease *); // expected-error {{no function template matches function template specialization 'templateFunction'}}

template<>
void templateReleaser(AntiRelease *); // expected-error {{no function template matches function template specialization 'templateReleaser'}}

template<>
void templateReleaser(ExplicitAntiRelease *) {} // expected-error {{no function template matches function template specialization 'templateReleaser'}}

template<>
void templateReleaser(__attribute__((ns_consumed)) ProRelease *); // no-warning
