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
