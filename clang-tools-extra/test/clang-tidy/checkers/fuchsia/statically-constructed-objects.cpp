// RUN: %check_clang_tidy %s fuchsia-statically-constructed-objects %t

// Trivial static is fine
static int i;

class ClassWithNoCtor {};

class ClassWithCtor {
public:
  ClassWithCtor(int Val) : Val(Val) {}
private:
  int Val;
};

class ClassWithConstexpr {
public:
  ClassWithConstexpr(int Val1, int Val2) : Val(Val1) {}
  constexpr ClassWithConstexpr(int Val) : Val(Val) {}

private:
  int Val;
};

ClassWithNoCtor A;
ClassWithConstexpr C(0);
ClassWithConstexpr E(0, 1);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: static objects are disallowed; if possible, use a constexpr constructor instead [fuchsia-statically-constructed-objects]
// CHECK-MESSAGES-NEXT:  ClassWithConstexpr E(0, 1);
ClassWithCtor G(0);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: static objects are disallowed; if possible, use a constexpr constructor instead [fuchsia-statically-constructed-objects]
// CHECK-MESSAGES-NEXT:  ClassWithCtor G(0);

static ClassWithNoCtor A2;
static ClassWithConstexpr C2(0);
static ClassWithConstexpr E2(0, 1);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: static objects are disallowed; if possible, use a constexpr constructor instead [fuchsia-statically-constructed-objects]
// CHECK-MESSAGES-NEXT:  static ClassWithConstexpr E2(0, 1);
static ClassWithCtor G2(0);
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: static objects are disallowed; if possible, use a constexpr constructor instead [fuchsia-statically-constructed-objects]
// CHECK-MESSAGES-NEXT:  static ClassWithCtor G2(0);

struct StructWithConstexpr { constexpr StructWithConstexpr(int Val) {} };
struct StructWithNoCtor {};
struct StructWithCtor { StructWithCtor(); };

StructWithNoCtor SNoCtor;
StructWithConstexpr SConstexpr(0);
StructWithCtor SCtor;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: static objects are disallowed; if possible, use a constexpr constructor instead [fuchsia-statically-constructed-objects]
// CHECK-MESSAGES-NEXT:  StructWithCtor SCtor;

static StructWithConstexpr SConstexpr2(0);
static StructWithNoCtor SNoCtor2;
static StructWithCtor SCtor2;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: static objects are disallowed; if possible, use a constexpr constructor instead [fuchsia-statically-constructed-objects]
// CHECK-MESSAGES-NEXT:  static StructWithCtor SCtor2;

extern StructWithCtor SCtor3;

class ClassWithStaticMember {
private:
  static StructWithNoCtor S;
};

ClassWithStaticMember Z();

class S {
  int Val;
public:
  constexpr S(int i) : Val(100 / i) {}
  int getVal() const { return Val; }
};

static S s1(1);
static S s2(0); 
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: static objects are disallowed; if possible, use a constexpr constructor instead [fuchsia-statically-constructed-objects]
// CHECK-MESSAGES-NEXT: static S s2(0);

extern int get_i();
static S s3(get_i());
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: static objects are disallowed; if possible, use a constexpr constructor instead [fuchsia-statically-constructed-objects]
// CHECK-MESSAGES-NEXT:  static S s3(get_i());

void f() {
  // Locally static is fine
  static int i;
  static ClassWithNoCtor A2;
  static ClassWithConstexpr C2(0);
  static ClassWithConstexpr E2(0, 1);
  static ClassWithCtor G2(0);
}
