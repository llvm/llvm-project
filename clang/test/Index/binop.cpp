// RUN: c-index-test -test-print-binops %s | FileCheck %s

struct C {
  int m;
};

void func(void) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
  int a, b;
  int C::*p = &C::m;

  C c;
  c.*p;

  C *pc;
  pc->*p;

  a *b;
  a / b;
  a % b;
  a + b;
  a - b;

  a << b;
  a >> b;

  a < b;
  a > b;

  a <= b;
  a >= b;
  a == b;
  a != b;

  a &b;
  a ^ b;
  a | b;

  a &&b;
  a || b;

  a = b;

  a *= b;
  a /= b;
  a %= b;
  a += b;
  a -= b;

  a <<= b;
  a >>= b;

  a &= b;
  a ^= b;
  a |= b;
  a, b;
#pragma clang diagnostic pop
}

// CHECK: BinaryOperator=.* BinOp=.* 1
// CHECK: BinaryOperator=->* BinOp=->* 2
// CHECK: BinaryOperator=* BinOp=* 3
// CHECK: BinaryOperator=/ BinOp=/ 4
// CHECK: BinaryOperator=% BinOp=% 5
// CHECK: BinaryOperator=+ BinOp=+ 6
// CHECK: BinaryOperator=- BinOp=- 7
// CHECK: BinaryOperator=<< BinOp=<< 8
// CHECK: BinaryOperator=>> BinOp=>> 9
// CHECK: BinaryOperator=< BinOp=< 11
// CHECK: BinaryOperator=> BinOp=> 12
// CHECK: BinaryOperator=<= BinOp=<= 13
// CHECK: BinaryOperator=>= BinOp=>= 14
// CHECK: BinaryOperator=== BinOp=== 15
// CHECK: BinaryOperator=!= BinOp=!= 16
// CHECK: BinaryOperator=& BinOp=& 17
// CHECK: BinaryOperator=^ BinOp=^ 18
// CHECK: BinaryOperator=| BinOp=| 19
// CHECK: BinaryOperator=&& BinOp=&& 20
// CHECK: BinaryOperator=|| BinOp=|| 21
// CHECK: BinaryOperator== BinOp== 22
// CHECK: CompoundAssignOperator=*= BinOp=*= 23
// CHECK: CompoundAssignOperator=/= BinOp=/= 24
// CHECK: CompoundAssignOperator=%= BinOp=%= 25
// CHECK: CompoundAssignOperator=+= BinOp=+= 26
// CHECK: CompoundAssignOperator=-= BinOp=-= 27
// CHECK: CompoundAssignOperator=<<= BinOp=<<= 28
// CHECK: CompoundAssignOperator=>>= BinOp=>>= 29
// CHECK: CompoundAssignOperator=&= BinOp=&= 30
// CHECK: CompoundAssignOperator=^= BinOp=^= 31
// CHECK: CompoundAssignOperator=|= BinOp=|= 32
// CHECK: BinaryOperator=, BinOp=, 33
