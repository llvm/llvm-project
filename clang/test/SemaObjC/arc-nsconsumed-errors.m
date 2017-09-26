// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -verify -fblocks -triple x86_64-apple-darwin10.0.0 %s
// rdar://10187884

typedef void (^blk)(id arg1, __attribute((ns_consumed)) id arg2);
typedef void (^blk1)(__attribute((ns_consumed))id arg1, __attribute((ns_consumed)) id arg2);
blk a = ^void (__attribute((ns_consumed)) id arg1, __attribute((ns_consumed)) id arg2){}; // expected-error {{incompatible block pointer types initializing}}

blk b = ^void (id arg1, __attribute((ns_consumed)) id arg2){};

blk c = ^void (__attribute((ns_consumed)) id arg1, __attribute((ns_consumed)) id arg2){}; // expected-error {{incompatible block pointer types initializing}}

blk d = ^void (id arg1, id arg2) {}; // expected-error {{incompatible block pointer types initializing}}

blk1 a1 = ^void (__attribute((ns_consumed)) id arg1, id arg2){}; // expected-error {{incompatible block pointer types initializing}}

blk1 b2 = ^void (id arg1, __attribute((ns_consumed)) id arg2){}; // expected-error {{incompatible block pointer types initializing}}

blk1 c3 = ^void (__attribute((ns_consumed)) id arg1, __attribute((ns_consumed)) id arg2){};

blk1 d4 = ^void (id arg1, id arg2) {}; // expected-error {{incompatible block pointer types initializing}}
