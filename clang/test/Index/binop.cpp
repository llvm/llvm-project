// RUN: c-index-test -test-print-binops -std=c++20 %s | FileCheck %s

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

struct D {
  D() = default;
  D& operator+(){return *this;}
  D& operator-(){return *this;}
  int& operator->*(int D::*i){return this->i;}
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
  D& operator&(const D&){return *this;}
  D& operator^(const D&){return *this;}
  D& operator|(const D&){return *this;}
  bool operator&&(const D&){return true;}
  bool operator||(const D&){return true;}
  D& operator=(const D&);
  D& operator*=(const D&){return *this;}
  D& operator/=(const D&){return *this;}
  D& operator%=(const D&){return *this;}
  D& operator+=(const D&){return *this;}
  D& operator-=(const D&){return *this;}
  D& operator<<=(const D&){return *this;}
  D& operator>>=(const D&){return *this;}
  D& operator&=(const D&){return *this;}
  D& operator^=(const D&){return *this;}
  D& operator|=(const D&){return *this;}
  D& operator,(const D&){return *this;}

  // Negative test of --/++
  D& operator++(int){return *this;};
  D& operator++(){return *this;};
  D& operator--(int){return *this;};
  D& operator--(){return *this;};
  void foo();
  int i;
};

// CHECK: CXXMethod=operator+:96:6 (Definition) BinOp= 0
// CHECK: CXXMethod=operator-:97:6 (Definition) BinOp= 0
// CHECK: CXXMethod=operator->*:98:8 (Definition) BinOp=->* 2
// CHECK: CXXMethod=operator*:99:6 (Definition) BinOp=* 3
// CHECK: CXXMethod=operator/:100:6 (Definition) BinOp=/ 4
// CHECK: CXXMethod=operator%:101:6 (Definition) BinOp=% 5
// CHECK: CXXMethod=operator+:102:6 (Definition) BinOp=+ 6
// CHECK: CXXMethod=operator-:103:6 (Definition) BinOp=- 7
// CHECK: CXXMethod=operator<<:104:6 (Definition) BinOp=<< 8
// CHECK: CXXMethod=operator>>:105:6 (Definition) BinOp=>> 9
// CHECK: CXXMethod=operator<:106:8 (Definition) BinOp=< 11
// CHECK: CXXMethod=operator>:107:8 (Definition) BinOp=> 12
// CHECK: CXXMethod=operator<=:108:8 (Definition) BinOp=<= 13
// CHECK: CXXMethod=operator>=:109:8 (Definition) BinOp=>= 14
// CHECK: CXXMethod=operator==:110:8 (Definition) BinOp=== 15
// CHECK: CXXMethod=operator!=:111:8 (Definition) BinOp=!= 16
// CHECK: CXXMethod=operator&:112:6 (Definition) BinOp=& 17
// CHECK: CXXMethod=operator^:113:6 (Definition) BinOp=^ 18
// CHECK: CXXMethod=operator|:114:6 (Definition) BinOp=| 19
// CHECK: CXXMethod=operator&&:115:8 (Definition) BinOp=&& 20
// CHECK: CXXMethod=operator||:116:8 (Definition) BinOp=|| 21
// CHECK: CXXMethod=operator=:117:6 (copy-assignment operator) BinOp== 22
// CHECK: CXXMethod=operator*=:118:6 (Definition) BinOp=*= 23
// CHECK: CXXMethod=operator/=:119:6 (Definition) BinOp=/= 24
// CHECK: CXXMethod=operator%=:120:6 (Definition) BinOp=%= 25
// CHECK: CXXMethod=operator+=:121:6 (Definition) BinOp=+= 26
// CHECK: CXXMethod=operator-=:122:6 (Definition) BinOp=-= 27
// CHECK: CXXMethod=operator<<=:123:6 (Definition) BinOp=<<= 28
// CHECK: CXXMethod=operator>>=:124:6 (Definition) BinOp=>>= 29
// CHECK: CXXMethod=operator&=:125:6 (Definition) BinOp=&= 30
// CHECK: CXXMethod=operator^=:126:6 (Definition) BinOp=^= 31
// CHECK: CXXMethod=operator|=:127:6 (Definition) BinOp=|= 32
// CHECK: CXXMethod=operator,:128:6 (Definition) BinOp=, 33
// CHECK: CXXMethod=operator++:131:6 (Definition) BinOp= 0
// CHECK: CXXMethod=operator++:132:6 (Definition) BinOp= 0
// CHECK: CXXMethod=operator--:133:6 (Definition) BinOp= 0
// CHECK: CXXMethod=operator--:134:6 (Definition) BinOp= 0
// CHECK: CXXMethod=foo:135:8 BinOp= 0


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
  
  +a;
  -a;
  #pragma clang diagnostic pop
}

// CHECK: FunctionDecl=func2:179:6 (Definition) BinOp= 0
// CHECK: CallExpr=D:95:3 BinOp= 0
// CHECK: CallExpr=D:95:3 BinOp= 0
// CHECK: CallExpr=operator->*:98:8 BinOp=->* 2
// CHECK: CallExpr=operator*:99:6 BinOp=* 3
// CHECK: CallExpr=operator/:100:6 BinOp=/ 4
// CHECK: CallExpr=operator%:101:6 BinOp=% 5
// CHECK: CallExpr=operator+:102:6 BinOp=+ 6
// CHECK: CallExpr=operator-:103:6 BinOp=- 7
// CHECK: CallExpr=operator<<:104:6 BinOp=<< 8
// CHECK: CallExpr=operator>>:105:6 BinOp=>> 9
// CHECK: CallExpr=operator<:106:8 BinOp=< 11
// CHECK: CallExpr=operator>:107:8 BinOp=> 12
// CHECK: CallExpr=operator<=:108:8 BinOp=<= 13
// CHECK: CallExpr=operator>=:109:8 BinOp=>= 14
// CHECK: CallExpr=operator==:110:8 BinOp=== 15
// CHECK: CallExpr=operator!=:111:8 BinOp=!= 16
// CHECK: CallExpr=operator&:112:6 BinOp=& 17
// CHECK: CallExpr=operator^:113:6 BinOp=^ 18
// CHECK: CallExpr=operator|:114:6 BinOp=| 19
// CHECK: CallExpr=operator&&:115:8 BinOp=&& 20
// CHECK: CallExpr=operator||:116:8 BinOp=|| 21
// CHECK: CallExpr=operator=:117:6 BinOp== 22
// CHECK: CallExpr=operator*=:118:6 BinOp=*= 23
// CHECK: CallExpr=operator/=:119:6 BinOp=/= 24
// CHECK: CallExpr=operator%=:120:6 BinOp=%= 25
// CHECK: CallExpr=operator+=:121:6 BinOp=+= 26
// CHECK: CallExpr=operator-=:122:6 BinOp=-= 27
// CHECK: CallExpr=operator<<=:123:6 BinOp=<<= 28
// CHECK: CallExpr=operator>>=:124:6 BinOp=>>= 29
// CHECK: CallExpr=operator&=:125:6 BinOp=&= 30
// CHECK: CallExpr=operator^=:126:6 BinOp=^= 31
// CHECK: CallExpr=operator|=:127:6 BinOp=|= 32
// CHECK: CallExpr=operator,:128:6 BinOp=, 33
// CHECK: CallExpr=operator++:131:6 BinOp= 0
// CHECK: CallExpr=operator++:132:6 BinOp= 0
// CHECK: CallExpr=operator--:133:6 BinOp= 0
// CHECK: CallExpr=operator--:134:6 BinOp= 0
// CHECK: CallExpr=operator+:96:6 BinOp= 0
// CHECK: CallExpr=operator-:97:6 BinOp= 0


struct E{
  int i;
};

int& operator->*(const E&, int E::*i);
E operator*(const E&, const E&);
E operator/(const E&, const E&);
E operator%(const E&, const E&);
E operator+(const E&, const E&);
E operator-(const E&, const E&);
E operator<<(const E&, const E&);
E operator>>(const E&, const E&);
bool operator<(const E&, const E&);
bool operator>(const E&, const E&);
bool operator<=(const E&, const E&);
bool operator>=(const E&, const E&);
bool operator==(const E&, const E&);
bool operator!=(const E&, const E&);
E operator&(const E&, const E&);
E operator^(const E&, const E&);
E operator|(const E&, const E&);
bool operator&&(const E&, const E&);
bool operator||(const E&, const E&);
E operator*=(const E&, const E&);
E operator/=(const E&, const E&);
E operator%=(const E&, const E&);
E operator+=(const E&, const E&);
E operator-=(const E&, const E&);
E operator<<=(const E&, const E&);
E operator>>=(const E&, const E&);
E operator&=(const E&, const E&);
E operator^=(const E&, const E&);
E operator|=(const E&, const E&);
E operator,(const E&, const E&);
E operator++(const E&a, int);
E operator++(const E&a );
E operator--(const E&a, int);
E operator--(const E&a);
void foo(const E&, const E&);
E operator+(const E&);
E operator-(const E&);

// CHECK: FunctionDecl=operator->*:285:6 BinOp=->* 2
// CHECK: FunctionDecl=operator*:286:3 BinOp=* 3
// CHECK: FunctionDecl=operator/:287:3 BinOp=/ 4
// CHECK: FunctionDecl=operator%:288:3 BinOp=% 5
// CHECK: FunctionDecl=operator+:289:3 BinOp=+ 6
// CHECK: FunctionDecl=operator-:290:3 BinOp=- 7
// CHECK: FunctionDecl=operator<<:291:3 BinOp=<< 8
// CHECK: FunctionDecl=operator>>:292:3 BinOp=>> 9
// CHECK: FunctionDecl=operator<:293:6 BinOp=< 11
// CHECK: FunctionDecl=operator>:294:6 BinOp=> 12
// CHECK: FunctionDecl=operator<=:295:6 BinOp=<= 13
// CHECK: FunctionDecl=operator>=:296:6 BinOp=>= 14
// CHECK: FunctionDecl=operator==:297:6 BinOp=== 15
// CHECK: FunctionDecl=operator!=:298:6 BinOp=!= 16
// CHECK: FunctionDecl=operator&:299:3 BinOp=& 17
// CHECK: FunctionDecl=operator^:300:3 BinOp=^ 18
// CHECK: FunctionDecl=operator|:301:3 BinOp=| 19
// CHECK: FunctionDecl=operator&&:302:6 BinOp=&& 20
// CHECK: FunctionDecl=operator||:303:6 BinOp=|| 21
// CHECK: FunctionDecl=operator*=:304:3 BinOp=*= 23
// CHECK: FunctionDecl=operator/=:305:3 BinOp=/= 24
// CHECK: FunctionDecl=operator%=:306:3 BinOp=%= 25
// CHECK: FunctionDecl=operator+=:307:3 BinOp=+= 26
// CHECK: FunctionDecl=operator-=:308:3 BinOp=-= 27
// CHECK: FunctionDecl=operator<<=:309:3 BinOp=<<= 28
// CHECK: FunctionDecl=operator>>=:310:3 BinOp=>>= 29
// CHECK: FunctionDecl=operator&=:311:3 BinOp=&= 30
// CHECK: FunctionDecl=operator^=:312:3 BinOp=^= 31
// CHECK: FunctionDecl=operator|=:313:3 BinOp=|= 32
// CHECK: FunctionDecl=operator,:314:3 BinOp=, 33
// CHECK: FunctionDecl=operator++:315:3 BinOp= 0
// CHECK: FunctionDecl=operator++:316:3 BinOp= 0
// CHECK: FunctionDecl=operator--:317:3 BinOp= 0
// CHECK: FunctionDecl=operator--:318:3 BinOp= 0
// CHECK: FunctionDecl=foo:319:6 BinOp= 0
// CHECK: FunctionDecl=operator+:320:3 BinOp= 0
// CHECK: FunctionDecl=operator-:321:3 BinOp= 0

void func3(void) {
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-value"
  E a, b;
  int E::*p = &E::i;
  
  E *pc;
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
  
  +a;
  -a;
  #pragma clang diagnostic pop
}

// CHECK: FunctionDecl=func3:361:6 (Definition) BinOp= 0
// CHECK: CallExpr=E:281:8 BinOp= 0
// CHECK: CallExpr=E:281:8 BinOp= 0
// CHECK: CallExpr=operator->*:285:6 BinOp=->* 2
// CHECK: CallExpr=operator*:286:3 BinOp=* 3
// CHECK: CallExpr=operator/:287:3 BinOp=/ 4
// CHECK: CallExpr=operator%:288:3 BinOp=% 5
// CHECK: CallExpr=operator+:289:3 BinOp=+ 6
// CHECK: CallExpr=operator-:290:3 BinOp=- 7
// CHECK: CallExpr=operator<<:291:3 BinOp=<< 8
// CHECK: CallExpr=operator>>:292:3 BinOp=>> 9
// CHECK: CallExpr=operator<:293:6 BinOp=< 11
// CHECK: CallExpr=operator>:294:6 BinOp=> 12
// CHECK: CallExpr=operator<=:295:6 BinOp=<= 13
// CHECK: CallExpr=operator>=:296:6 BinOp=>= 14
// CHECK: CallExpr=operator==:297:6 BinOp=== 15
// CHECK: CallExpr=operator!=:298:6 BinOp=!= 16
// CHECK: CallExpr=operator&:299:3 BinOp=& 17
// CHECK: CallExpr=operator^:300:3 BinOp=^ 18
// CHECK: CallExpr=operator|:301:3 BinOp=| 19
// CHECK: CallExpr=operator&&:302:6 BinOp=&& 20
// CHECK: CallExpr=operator||:303:6 BinOp=|| 21
// CHECK: CallExpr=operator=:281:8 BinOp== 22
// CHECK: CallExpr=operator*=:304:3 BinOp=*= 23
// CHECK: CallExpr=operator/=:305:3 BinOp=/= 24
// CHECK: CallExpr=operator%=:306:3 BinOp=%= 25
// CHECK: CallExpr=operator+=:307:3 BinOp=+= 26
// CHECK: CallExpr=operator-=:308:3 BinOp=-= 27
// CHECK: CallExpr=operator<<=:309:3 BinOp=<<= 28
// CHECK: CallExpr=operator>>=:310:3 BinOp=>>= 29
// CHECK: CallExpr=operator&=:311:3 BinOp=&= 30
// CHECK: CallExpr=operator^=:312:3 BinOp=^= 31
// CHECK: CallExpr=operator|=:313:3 BinOp=|= 32
// CHECK: CallExpr=operator,:314:3 BinOp=, 33
// CHECK: CallExpr=operator++:315:3 BinOp= 0
// CHECK: CallExpr=operator++:316:3 BinOp= 0
// CHECK: CallExpr=operator--:317:3 BinOp= 0
// CHECK: CallExpr=operator--:318:3 BinOp= 0
// CHECK: CallExpr=operator+:320:3 BinOp= 0
// CHECK: CallExpr=operator-:321:3 BinOp= 0


struct space1{
  int operator<=>(const space1&) const;
  bool operator==(const space1&) const;
};

void func4(){
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-value"
  space1 s1, s2;
  s1 <=> s2;
  s1 < s2;
  s1 > s2;
  s1 <= s2;
  s1 >= s2;
  s1 == s2;
  s1 != s2;
#pragma clang diagnostic pop
}

// CHECK: FunctionDecl=func4:468:6 (Definition) BinOp= 0
// CHECK: CallExpr=space1:463:8 BinOp= 0
// CHECK: CallExpr=space1:463:8 BinOp= 0
// CHECK: CallExpr=operator<=>:464:7 BinOp=<=> 10
// CHECK: BinaryOperator=< BinOp=< 11
// CHECK: CallExpr=operator<=>:464:7 BinOp=<=> 10
// CHECK: BinaryOperator=> BinOp=> 12
// CHECK: CallExpr=operator<=>:464:7 BinOp=<=> 10
// CHECK: BinaryOperator=<= BinOp=<= 13
// CHECK: CallExpr=operator<=>:464:7 BinOp=<=> 10
// CHECK: BinaryOperator=>= BinOp=>= 14
// CHECK: CallExpr=operator<=>:464:7 BinOp=<=> 10
// CHECK: CallExpr=operator==:465:8 BinOp=== 15
// CHECK: CallExpr=operator==:465:8 BinOp=== 15

struct space2{};
int operator<=>(const space2&, const space2&);
bool operator ==(const space2 &, const space2&);
void func5(){
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-value"
  space2 s1, s2;
  s1 <=> s2;
  s1 < s2;
  s1 > s2;
  s1 <= s2;
  s1 >= s2;
  s1 == s2;
  s1 != s2;
#pragma clang diagnostic pop
}

// CHECK: FunctionDecl=func5:500:6 (Definition) BinOp= 0
// CHECK: CallExpr=space2:497:8 BinOp= 0
// CHECK: CallExpr=space2:497:8 BinOp= 0
// CHECK: CallExpr=operator<=>:498:5 BinOp=<=> 10
// CHECK: BinaryOperator=< BinOp=< 11
// CHECK: CallExpr=operator<=>:498:5 BinOp=<=> 10
// CHECK: BinaryOperator=> BinOp=> 12
// CHECK: CallExpr=operator<=>:498:5 BinOp=<=> 10
// CHECK: BinaryOperator=<= BinOp=<= 13
// CHECK: CallExpr=operator<=>:498:5 BinOp=<=> 10
// CHECK: BinaryOperator=>= BinOp=>= 14
// CHECK: CallExpr=operator<=>:498:5 BinOp=<=> 10
// CHECK: CallExpr=operator==:499:6 BinOp=== 15
// CHECK: CallExpr=operator==:499:6 BinOp=== 15
