// RUN: %clang_cc1 %s -fsyntax-only -verify -fblocks

@class NSObject;
typedef void (^block1_t)(int arg);
typedef void (^block2_t)(block1_t arg);
typedef void (^block3_t)(NSObject *arg);
typedef void (^block4_t)(id arg);

void fn(block4_t arg); // expected-note {{passing argument to parameter 'arg' here}}

void another_fn(block2_t arg);

int main(void) {
    block1_t b1;
    block2_t b2;
    block3_t b3;
    block3_t b4;
    block4_t b5;

    fn(b1);  // expected-error {{incompatible block pointer types passing 'block1_t' (aka 'void (^)(int)') to parameter of type 'block4_t' (aka 'void (^)(id)')}}
    fn(b2);  // must succeed: block1_t *is* compatible with id
    fn(b3);  // succeeds: NSObject* compatible with id
    fn(b4);  // succeeds: id compatible with id

    another_fn(b5);
}
