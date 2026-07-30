// Test is line- and column-sensitive. Run lines are below.

struct X {
  X();
  X(int);
  X(int, int);
  X(const X&);
};

X getX(int value) { 
  switch (value) {
  case 1: return X(value);
  case 2: return X(value, value);
  case 3: return (X)value;
  default: break;
  }
  return X();
}

struct Y {
  int member;

  X getX();
};

X Y::getX() {
  return member;
}

struct YDerived : Y {
  X getAnotherX() { return member; }
};

void test() {
  X foo;

  try {
  } catch (X e) {
    X x;
  }

  struct LocalS {
    void meth() {
      int x;
      ++x;
    }
  };
}

template <bool (*tfn)(X*)>
struct TS {
  void foo();
};

template <bool (*tfn)(X*)>
void TS<tfn>::foo() {}

template <typename T>
class TC {
  void init();
};

template<> void TC<char>::init();

#define EXTERN_TEMPLATE(...) extern template __VA_ARGS__;
EXTERN_TEMPLATE(class TC<char>)

class A {
  A();
  virtual ~A();

  // Assignment operators
  A& operator=(const A&);
  A& operator=(A&&) noexcept;

  // Unary operators
  A    operator+() const;
  A    operator-() const;
  A    operator~() const;
  A    operator*() const;
  A    operator&() const;
  bool operator!() const;

  // (pre-|post-) increment and decrement
  A& operator++();
  A& operator--();
  A  operator++(int);
  A  operator--(int);

  // Arithmetic operators
  A operator+(const A&) const;
  A operator-(const A&) const;
  A operator*(const A&) const;
  A operator/(const A&) const;
  A operator%(const A&) const;
  A operator&(const A&) const;
  A operator|(const A&) const;
  A operator^(const A&) const;

  A operator<<(const A&) const;
  A operator>>(const A&) const;

  // Arithmetic-assignment operators
  A& operator+=(const A&);
  A& operator-=(const A&);
  A& operator*=(const A&);
  A& operator/=(const A&);
  A& operator%=(const A&);
  A& operator&=(const A&);
  A& operator|=(const A&);
  A& operator^=(const A&);

  A& operator<<=(const A&);
  A& operator>>=(const A&);

  // Logical operators
  bool operator<(const A&) const;
  bool operator>(const A&) const;

  bool operator&&(const A&) const;
  bool operator||(const A&) const;
  bool operator<=(const A&) const;
  bool operator>=(const A&) const;
  bool operator!=(const A&) const;
  bool operator==(const A&) const;

  // Special operators
  A& operator[](unsigned long long);
  A* operator->();
  A  operator()(unsigned, int) const;

  explicit operator bool() const;
};

struct TestColl {
  int* begin();
  int* end();
};

void test(TestColl coll) {
  for (auto lv : coll) {
    (void)lv;
  }
}

const int operator""_toint(unsigned long long val) { return int(val); }

// noexcept specifications
void f_noexcept() noexcept;
template <class T> void f_computed_noexcept(T t) noexcept(noexcept(t+t));
void f_dynamic_noexcept_none() throw();
void f_dynamic_noexcept() throw(int);
void f_dynamic_noexcept_any() throw(...);
void f_computed_noexcept_true() noexcept(true);
void f_computed_noexcept_false() noexcept(false);

enum EnumType { Enumerator };
struct Z {
    EnumType e = Enumerator;
};

// RUN: c-index-test -cursor-at=%s:6:4 %s | FileCheck -check-prefix=CHECK-COMPLETION-1 %s
// CHECK-COMPLETION-1: CXXConstructor=X:6:3
// CHECK-COMPLETION-1-NEXT: Completion string: {TypedText X}{LeftParen (}{Placeholder int}{Comma , }{Placeholder int}{RightParen )}

// RUN: c-index-test -cursor-at=%s:31:16 %s | FileCheck -check-prefix=CHECK-COMPLETION-2 %s
// CHECK-COMPLETION-2: CXXMethod=getAnotherX:31:5 (Definition)
// CHECK-COMPLETION-2-NEXT: Completion string: {ResultType X}{TypedText getAnotherX}{LeftParen (}{RightParen )}

// RUN: c-index-test -cursor-at=%s:12:20 %s | FileCheck -check-prefix=CHECK-VALUE-REF %s
// RUN: c-index-test -cursor-at=%s:13:21 %s | FileCheck -check-prefix=CHECK-VALUE-REF %s
// RUN: c-index-test -cursor-at=%s:13:28 %s | FileCheck -check-prefix=CHECK-VALUE-REF %s
// RUN: c-index-test -cursor-at=%s:14:23 %s | FileCheck -check-prefix=CHECK-VALUE-REF %s
// CHECK-VALUE-REF: DeclRefExpr=value:10:12

// RUN: c-index-test -cursor-at=%s:12:18 %s | FileCheck -check-prefix=CHECK-CONSTRUCTOR1 %s
// RUN: c-index-test -cursor-at=%s:13:18 %s | FileCheck -check-prefix=CHECK-CONSTRUCTOR2 %s
// RUN: c-index-test -cursor-at=%s:14:19 %s | FileCheck -check-prefix=CHECK-CONSTRUCTOR1 %s
// RUN: c-index-test -cursor-at=%s:17:10 %s | FileCheck -check-prefix=CHECK-CONSTRUCTOR3 %s
// CHECK-TYPE-REF: TypeRef=struct X:3:8
// CHECK-CONSTRUCTOR1: CallExpr=X:5:3
// CHECK-CONSTRUCTOR2: CallExpr=X:6:3
// CHECK-CONSTRUCTOR3: CallExpr=X:4:3

// RUN: c-index-test -cursor-at=%s:23:3 %s | FileCheck -check-prefix=CHECK-RETTYPE %s
// RUN: c-index-test -cursor-at=%s:26:1 %s | FileCheck -check-prefix=CHECK-RETTYPE %s
// CHECK-RETTYPE: TypeRef=struct X:3:8

// RUN: c-index-test -cursor-at=%s:23:7 %s | FileCheck -check-prefix=CHECK-MEMFUNC-DECL %s
// CHECK-MEMFUNC-DECL: CXXMethod=getX:23:5
// RUN: c-index-test -cursor-at=%s:26:7 %s | FileCheck -check-prefix=CHECK-MEMFUNC-DEF %s
// CHECK-MEMFUNC-DEF: CXXMethod=getX:26:6

// RUN: c-index-test -cursor-at=%s:26:3 %s | FileCheck -check-prefix=CHECK-TYPEREF-Y %s
// CHECK-TYPEREF-Y: TypeRef=struct Y:20:8

// RUN: c-index-test -cursor-at=%s:27:10 %s | FileCheck -check-prefix=CHECK-IMPLICIT-MEMREF %s
// RUN: c-index-test -cursor-at=%s:31:28 %s | FileCheck -check-prefix=CHECK-IMPLICIT-MEMREF %s
// CHECK-IMPLICIT-MEMREF: MemberRefExpr=member:21:7

// RUN: c-index-test -cursor-at=%s:35:5 %s | FileCheck -check-prefix=CHECK-DECL %s
// CHECK-DECL: VarDecl=foo:35:5

// RUN: c-index-test -cursor-at=%s:21:3 %s | FileCheck -check-prefix=CHECK-MEMBER %s
// CHECK-MEMBER: FieldDecl=member:21:7 (Definition)
// CHECK-MEMBER-NEXT: Completion string: {ResultType int}{TypedText member}

// RUN: c-index-test -cursor-at=%s:38:12 -cursor-at=%s:39:5 %s | FileCheck -check-prefix=CHECK-CXXCATCH %s
// CHECK-CXXCATCH: TypeRef=struct X:3:8
// CHECK-CXXCATCH-NEXT: TypeRef=struct X:3:8

// RUN: c-index-test -test-load-source-usrs local %s | FileCheck -check-prefix=CHECK-USR %s
// CHECK-USR: get-cursor.cpp c:get-cursor.cpp@472@F@test#@e Extent=[38:12 - 38:15]
// CHECK-USR: get-cursor.cpp c:get-cursor.cpp@483@F@test#@x Extent=[39:5 - 39:8]

// RUN: c-index-test -cursor-at=%s:45:9 %s | FileCheck -check-prefix=CHECK-LOCALCLASS %s
// CHECK-LOCALCLASS: 45:9 DeclRefExpr=x:44:11 Extent=[45:9 - 45:10] Spelling=x ([45:9 - 45:10])

// RUN: c-index-test -cursor-at=%s:50:23 -cursor-at=%s:55:23 %s | FileCheck -check-prefix=CHECK-TEMPLPARAM %s
// CHECK-TEMPLPARAM: 50:23 TypeRef=struct X:3:8 Extent=[50:23 - 50:24] Spelling=struct X ([50:23 - 50:24])
// CHECK-TEMPLPARAM: 55:23 TypeRef=struct X:3:8 Extent=[55:23 - 55:24] Spelling=struct X ([55:23 - 55:24])

// RUN: c-index-test -cursor-at=%s:66:23 %s | FileCheck -check-prefix=CHECK-TEMPLSPEC %s
// CHECK-TEMPLSPEC: 66:23 ClassDecl=TC:66:23 (Definition) [Specialization of TC:59:7] [Template arg 0: kind: 1, type: char] Extent=[66:1 - 66:31] Spelling=TC ([66:23 - 66:25])

// RUN: c-index-test -cursor-at=%s:69:3 -cursor-at=%s:70:11 -cursor-at=%s:73:6 -cursor-at=%s:74:6 -cursor-at=%s:77:8 -cursor-at=%s:78:8 -cursor-at=%s:79:8 -cursor-at=%s:80:8 -cursor-at=%s:81:8 -cursor-at=%s:82:8 -cursor-at=%s:85:6 -cursor-at=%s:86:6 -cursor-at=%s:87:6 -cursor-at=%s:88:6 -cursor-at=%s:91:5 -cursor-at=%s:92:5 -cursor-at=%s:93:5 -cursor-at=%s:94:5 -cursor-at=%s:95:5 -cursor-at=%s:96:5 -cursor-at=%s:97:5 -cursor-at=%s:98:5 -cursor-at=%s:100:5 -cursor-at=%s:101:5 -cursor-at=%s:104:6 -cursor-at=%s:105:6 -cursor-at=%s:106:6 -cursor-at=%s:107:6 -cursor-at=%s:108:6 -cursor-at=%s:109:6 -cursor-at=%s:110:6 -cursor-at=%s:111:6 -cursor-at=%s:113:6 -cursor-at=%s:114:6 -cursor-at=%s:117:8 -cursor-at=%s:118:8 -cursor-at=%s:120:8 -cursor-at=%s:121:8 -cursor-at=%s:122:8 -cursor-at=%s:123:8 -cursor-at=%s:124:8 -cursor-at=%s:125:8 -cursor-at=%s:128:6 -cursor-at=%s:129:6 -cursor-at=%s:130:6 -cursor-at=%s:132:3 -cursor-at=%s:146:15 -cursor-at=%s:149:6 -cursor-at=%s:150:25 -cursor-at=%s:151:6 -cursor-at=%s:152:6 -cursor-at=%s:153:6 -cursor-at=%s:154:6 -cursor-at=%s:155:6 -std=c++11 %s | FileCheck -check-prefix=CHECK-SPELLING %s
// CHECK-SPELLING: 69:3 CXXConstructor=A:69:3 (default constructor) Extent=[69:3 - 69:6] Spelling=A ([69:3 - 69:4])
// CHECK-SPELLING: 70:11 CXXDestructor=~A:70:11 (virtual) Extent=[70:3 - 70:15] Spelling=~A ([70:11 - 70:13])
// CHECK-SPELLING: 73:6 CXXMethod=operator=:73:6 (copy-assignment operator) Extent=[73:3 - 73:25] Spelling=operator= ([73:6 - 73:15])
// CHECK-SPELLING: 74:6 CXXMethod=operator=:74:6 (move-assignment operator) (noexcept) Extent=[74:3 - 74:29] Spelling=operator= ([74:6 - 74:15])
// CHECK-SPELLING: 77:8 CXXMethod=operator+:77:8 (const) Extent=[77:3 - 77:25] Spelling=operator+ ([77:8 - 77:17])
// CHECK-SPELLING: 78:8 CXXMethod=operator-:78:8 (const) Extent=[78:3 - 78:25] Spelling=operator- ([78:8 - 78:17])
// CHECK-SPELLING: 79:8 CXXMethod=operator~:79:8 (const) Extent=[79:3 - 79:25] Spelling=operator~ ([79:8 - 79:17])
// CHECK-SPELLING: 80:8 CXXMethod=operator*:80:8 (const) Extent=[80:3 - 80:25] Spelling=operator* ([80:8 - 80:17])
// CHECK-SPELLING: 81:8 CXXMethod=operator&:81:8 (const) Extent=[81:3 - 81:25] Spelling=operator& ([81:8 - 81:17])
// CHECK-SPELLING: 82:8 CXXMethod=operator!:82:8 (const) Extent=[82:3 - 82:25] Spelling=operator! ([82:8 - 82:17])
// CHECK-SPELLING: 85:6 CXXMethod=operator++:85:6 Extent=[85:3 - 85:18] Spelling=operator++ ([85:6 - 85:16])
// CHECK-SPELLING: 86:6 CXXMethod=operator--:86:6 Extent=[86:3 - 86:18] Spelling=operator-- ([86:6 - 86:16])
// CHECK-SPELLING: 87:6 CXXMethod=operator++:87:6 Extent=[87:3 - 87:21] Spelling=operator++ ([87:6 - 87:16])
// CHECK-SPELLING: 88:6 CXXMethod=operator--:88:6 Extent=[88:3 - 88:21] Spelling=operator-- ([88:6 - 88:16])
// CHECK-SPELLING: 91:5 CXXMethod=operator+:91:5 (const) Extent=[91:3 - 91:30] Spelling=operator+ ([91:5 - 91:14])
// CHECK-SPELLING: 92:5 CXXMethod=operator-:92:5 (const) Extent=[92:3 - 92:30] Spelling=operator- ([92:5 - 92:14])
// CHECK-SPELLING: 93:5 CXXMethod=operator*:93:5 (const) Extent=[93:3 - 93:30] Spelling=operator* ([93:5 - 93:14])
// CHECK-SPELLING: 94:5 CXXMethod=operator/:94:5 (const) Extent=[94:3 - 94:30] Spelling=operator/ ([94:5 - 94:14])
// CHECK-SPELLING: 95:5 CXXMethod=operator%:95:5 (const) Extent=[95:3 - 95:30] Spelling=operator% ([95:5 - 95:14])
// CHECK-SPELLING: 96:5 CXXMethod=operator&:96:5 (const) Extent=[96:3 - 96:30] Spelling=operator& ([96:5 - 96:14])
// CHECK-SPELLING: 97:5 CXXMethod=operator|:97:5 (const) Extent=[97:3 - 97:30] Spelling=operator| ([97:5 - 97:14])
// CHECK-SPELLING: 98:5 CXXMethod=operator^:98:5 (const) Extent=[98:3 - 98:30] Spelling=operator^ ([98:5 - 98:14])
// CHECK-SPELLING: 100:5 CXXMethod=operator<<:100:5 (const) Extent=[100:3 - 100:31] Spelling=operator<< ([100:5 - 100:15])
// CHECK-SPELLING: 101:5 CXXMethod=operator>>:101:5 (const) Extent=[101:3 - 101:31] Spelling=operator>> ([101:5 - 101:15])
// CHECK-SPELLING: 104:6 CXXMethod=operator+=:104:6 Extent=[104:3 - 104:26] Spelling=operator+= ([104:6 - 104:16])
// CHECK-SPELLING: 105:6 CXXMethod=operator-=:105:6 Extent=[105:3 - 105:26] Spelling=operator-= ([105:6 - 105:16])
// CHECK-SPELLING: 106:6 CXXMethod=operator*=:106:6 Extent=[106:3 - 106:26] Spelling=operator*= ([106:6 - 106:16])
// CHECK-SPELLING: 107:6 CXXMethod=operator/=:107:6 Extent=[107:3 - 107:26] Spelling=operator/= ([107:6 - 107:16])
// CHECK-SPELLING: 108:6 CXXMethod=operator%=:108:6 Extent=[108:3 - 108:26] Spelling=operator%= ([108:6 - 108:16])
// CHECK-SPELLING: 109:6 CXXMethod=operator&=:109:6 Extent=[109:3 - 109:26] Spelling=operator&= ([109:6 - 109:16])
// CHECK-SPELLING: 110:6 CXXMethod=operator|=:110:6 Extent=[110:3 - 110:26] Spelling=operator|= ([110:6 - 110:16])
// CHECK-SPELLING: 111:6 CXXMethod=operator^=:111:6 Extent=[111:3 - 111:26] Spelling=operator^= ([111:6 - 111:16])
// CHECK-SPELLING: 113:6 CXXMethod=operator<<=:113:6 Extent=[113:3 - 113:27] Spelling=operator<<= ([113:6 - 113:17])
// CHECK-SPELLING: 114:6 CXXMethod=operator>>=:114:6 Extent=[114:3 - 114:27] Spelling=operator>>= ([114:6 - 114:17])
// CHECK-SPELLING: 117:8 CXXMethod=operator<:117:8 (const) Extent=[117:3 - 117:33] Spelling=operator< ([117:8 - 117:17])
// CHECK-SPELLING: 118:8 CXXMethod=operator>:118:8 (const) Extent=[118:3 - 118:33] Spelling=operator> ([118:8 - 118:17])
// CHECK-SPELLING: 120:8 CXXMethod=operator&&:120:8 (const) Extent=[120:3 - 120:34] Spelling=operator&& ([120:8 - 120:18])
// CHECK-SPELLING: 121:8 CXXMethod=operator||:121:8 (const) Extent=[121:3 - 121:34] Spelling=operator|| ([121:8 - 121:18])
// CHECK-SPELLING: 122:8 CXXMethod=operator<=:122:8 (const) Extent=[122:3 - 122:34] Spelling=operator<= ([122:8 - 122:18])
// CHECK-SPELLING: 123:8 CXXMethod=operator>=:123:8 (const) Extent=[123:3 - 123:34] Spelling=operator>= ([123:8 - 123:18])
// CHECK-SPELLING: 124:8 CXXMethod=operator!=:124:8 (const) Extent=[124:3 - 124:34] Spelling=operator!= ([124:8 - 124:18])
// CHECK-SPELLING: 125:8 CXXMethod=operator==:125:8 (const) Extent=[125:3 - 125:34] Spelling=operator== ([125:8 - 125:18])
// CHECK-SPELLING: 128:6 CXXMethod=operator[]:128:6 Extent=[128:3 - 128:36] Spelling=operator[] ([128:6 - 128:16])
// CHECK-SPELLING: 129:6 CXXMethod=operator->:129:6 Extent=[129:3 - 129:18] Spelling=operator-> ([129:6 - 129:16])
// CHECK-SPELLING: 130:6 CXXMethod=operator():130:6 (const) Extent=[130:3 - 130:37] Spelling=operator() ([130:6 - 130:16])
// CHECK-SPELLING: 132:12 CXXConversion=operator bool:132:12 (const) (explicit) Extent=[132:3 - 132:33] Spelling=operator bool ([132:12 - 132:25])
// CHECK-SPELLING: 146:11 FunctionDecl=operator""_toint:146:11 (Definition) Extent=[146:1 - 146:72] Spelling=operator""_toint ([146:11 - 146:27])
// CHECK-SPELLING: 149:6 FunctionDecl=f_noexcept:149:6 (noexcept) Extent=[149:1 - 149:27] Spelling=f_noexcept ([149:6 - 149:16])
// CHECK-SPELLING: 150:25 FunctionTemplate=f_computed_noexcept:150:25 (computed-noexcept) Extent=[150:1 - 150:73] Spelling=f_computed_noexcept ([150:25 - 150:44])
// CHECK-SPELLING: 151:6 FunctionDecl=f_dynamic_noexcept_none:151:6 (noexcept dynamic none) Extent=[151:1 - 151:39] Spelling=f_dynamic_noexcept_none ([151:6 - 151:29])
// CHECK-SPELLING: 152:6 FunctionDecl=f_dynamic_noexcept:152:6 (noexcept dynamic) Extent=[152:1 - 152:37] Spelling=f_dynamic_noexcept ([152:6 - 152:24])
// CHECK-SPELLING: 153:6 FunctionDecl=f_dynamic_noexcept_any:153:6 (noexcept dynamic any) Extent=[153:1 - 153:41] Spelling=f_dynamic_noexcept_any ([153:6 - 153:28])
// CHECK-SPELLING: 154:6 FunctionDecl=f_computed_noexcept_true:154:6 (computed-noexcept) Extent=[154:1 - 154:47] Spelling=f_computed_noexcept_true ([154:6 - 154:30])
// CHECK-SPELLING: 155:6 FunctionDecl=f_computed_noexcept_false:155:6 (computed-noexcept) Extent=[155:1 - 155:49] Spelling=f_computed_noexcept_false ([155:6 - 155:31])

// RUN: c-index-test -cursor-at=%s:141:13 -cursor-at=%s:141:18 -cursor-at=%s:142:11 -std=c++11 %s | FileCheck -check-prefix=CHECK-FORRANGE %s
// CHECK-FORRANGE: 141:13 VarDecl=lv:141:13 (Definition) Extent=[141:8 - 141:17] Spelling=lv ([141:13 - 141:15])
// CHECK-FORRANGE: 141:18 DeclRefExpr=coll:140:20 Extent=[141:18 - 141:22] Spelling=coll ([141:18 - 141:22])
// CHECK-FORRANGE: 142:11 DeclRefExpr=lv:141:13 Extent=[142:11 - 142:13] Spelling=lv ([142:11 - 142:13])

// RUN: c-index-test -cursor-at=%s:159:18 -std=c++11 %s | FileCheck -check-prefix=CHECK-INCLASSINITIALIZER %s
// CHECK-INCLASSINITIALIZER: 159:18 DeclRefExpr=Enumerator:157:17 Extent=[159:18 - 159:28] Spelling=Enumerator ([159:18 - 159:28])

