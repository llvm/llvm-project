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

// CHECK: CallExpr=C:3:8 BinOp= 0
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

struct D {
  D() = default;
  D& operator+(){return *this;}
  D& operator-(){return *this;}
  D& operator*(const D&){return *this;}
  D& operator/(const D&){return *this;}
  D& operator%(const D&){return *this;}
  D& operator+(const D&){return *this;}
  D& operator-(const D&){return *this;}

  D& operator<<(const D&){return *this;}
  D& operator>>(const D&){return *this;}

  bool operator<(const D&){return true;}
  bool operator>(const D&){return true;}
  bool operator<=(const D&){return true;}
  bool operator>=(const D&){return true;}
  bool operator==(const D&){return true;}
  bool operator!=(const D&){return true;}

  D& operator|(const D&){return *this;}
  D& operator&(const D&){return *this;}
  D& operator^(const D&){return *this;}

  bool operator&&(const D&){return true;}
  bool operator||(const D&){return true;}

  D& operator+=(const D&){return *this;}
  D& operator-=(const D&){return *this;}
  D& operator*=(const D&){return *this;}
  D& operator/=(const D&){return *this;}
  D& operator%=(const D&){return *this;}
  D& operator&=(const D&){return *this;}
  D& operator|=(const D&){return *this;}
  D& operator^=(const D&){return *this;}
  D& operator<<=(const D&){return *this;}
  D& operator>>=(const D&){return *this;}
  D& operator,(const D&){return *this;}

  int& operator->*(int D::*i){return this->i;}

  // Negative test of --/++
  D& operator++(){return *this;};
  D& operator++(int){return *this;};
  D& operator--(){return *this;};
  D& operator--(int){return *this;};

  int i;
};

void func2(void) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-value"
  D a, b;
  int D::*p = &D::i;

  D *pc;
  a->*p;

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

// Negative test
  a++;
  ++a;
  a--;
  --a;
  #pragma clang diagnostic pop
}

// CHECK: CallExpr=D:96:3 BinOp= 0
// CHECK: CallExpr=D:96:3 BinOp= 0
// CHECK: CallExpr=operator->*:134:8 BinOp=->* 2
// CHECK: CallExpr=operator*:99:6 BinOp=* 3
// CHECK: CallExpr=operator/:100:6 BinOp=/ 4
// CHECK: CallExpr=operator%:101:6 BinOp=% 5
// CHECK: CallExpr=operator+:102:6 BinOp=+ 6
// CHECK: CallExpr=operator-:103:6 BinOp=- 7
// CHECK: CallExpr=operator<<:105:6 BinOp=<< 8
// CHECK: CallExpr=operator>>:106:6 BinOp=>> 9
// CHECK: CallExpr=operator<:108:8 BinOp=< 11
// CHECK: CallExpr=operator>:109:8 BinOp=> 12
// CHECK: CallExpr=operator<=:110:8 BinOp=<= 13
// CHECK: CallExpr=operator>=:111:8 BinOp=>= 14
// CHECK: CallExpr=operator==:112:8 BinOp=== 15
// CHECK: CallExpr=operator!=:113:8 BinOp=!= 16
// CHECK: CallExpr=operator&:116:6 BinOp=& 17
// CHECK: CallExpr=operator^:117:6 BinOp=^ 18
// CHECK: CallExpr=operator|:115:6 BinOp=| 19
// CHECK: CallExpr=operator&&:119:8 BinOp=&& 20
// CHECK: CallExpr=operator||:120:8 BinOp=|| 21
// CHECK: CallExpr=operator=:95:8 BinOp== 22
// CHECK: CallExpr=operator*=:124:6 BinOp=*= 23
// CHECK: CallExpr=operator/=:125:6 BinOp=/= 24
// CHECK: CallExpr=operator%=:126:6 BinOp=%= 25
// CHECK: CallExpr=operator+=:122:6 BinOp=+= 26
// CHECK: CallExpr=operator-=:123:6 BinOp=-= 27
// CHECK: CallExpr=operator<<=:130:6 BinOp=<<= 28
// CHECK: CallExpr=operator>>=:131:6 BinOp=>>= 29
// CHECK: CallExpr=operator&=:127:6 BinOp=&= 30
// CHECK: CallExpr=operator^=:129:6 BinOp=^= 31
// CHECK: CallExpr=operator|=:128:6 BinOp=|= 32
// CHECK: CallExpr=operator,:132:6 BinOp=, 33
// CHECK: CallExpr=operator++:138:6 BinOp= 0
// CHECK: CallExpr=operator++:137:6 BinOp= 0
// CHECK: CallExpr=operator--:140:6 BinOp= 0
// CHECK: CallExpr=operator--:139:6 BinOp= 0
