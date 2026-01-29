namespace foo {
  int x;
  void bar(int z);
}
namespace bar {
  typedef int QType;
  void bar(QType z);
}

class ClsA {
public:
  int a, b;
  ClsA(int A, int B) : a(A), b(B) {}
};

namespace foo {
  class ClsB : public ClsA {
  public:
    ClsB() : ClsA(1, 2) {}
    int result() const;
  };
}

int foo::ClsB::result() const {
  return a + b;
}

namespace {
  class ClsC : public foo::ClsB {};
  int w;
}

int z;

namespace foo { namespace taz {
  int x;
  static inline int add(int a, int b) { return a + b; }
  void sub(int a, int b);
}
}

namespace foo { namespace taz {
  class ClsD : public foo::ClsB {
  public:
    ClsD& operator=(int x) { a = x; return *this; }
    ClsD& operator=(double x) { a = (int) x; return *this; }
    ClsD& operator=(const ClsD &x) { a = x.a; return *this; }
    static int qux();
    static int uz(int z, ...);
    bool operator==(const ClsD &x) const { return a == x.a; }
  };
}}

extern "C" {
  void rez(int a, int b);
}

namespace foo_alias = foo;

using namespace foo;

namespace foo_alias2 = foo;

using foo::ClsB;

namespace foo_alias3 = foo;

namespace {
class RDar9371763_Foo {
public:
  void bar();
};
}

void RDar9371763_Foo::bar() {}

void rdar9371763() {
  RDar9371763_Foo foo;
  foo.bar();
}

template <class T> typename T::A someTemplFn() {}
template <class T> typename T::B someTemplFn() {}
template <class T> int someTemplFn() {}

void funWithChar(char c) {}
void funWithChar(unsigned char c) {}
void funWithChar(signed char c) {}

struct { int x; } embedS1;
struct { int x; } embedS2;

template <typename T>
class TC1 {
  void meth(TC1);
};

typedef __attribute__((__ext_vector_type__(3))) float float3;

typedef float __m128 __attribute__((__vector_size__(16)));

float3 vectorOverload(float3 f);

__m128 vectorOverload(__m128 f);

// RUN: c-index-test -test-load-source-usrs all -fno-delayed-template-parsing %s | FileCheck %s
// CHECK: usrs.cpp c:@N@foo Extent=[1:1 - 4:2]
// CHECK: usrs.cpp c:@N@foo@x Extent=[2:3 - 2:8]
// CHECK: usrs.cpp c:@N@foo@F@bar#I# Extent=[3:3 - 3:18]
// CHECK: usrs.cpp c:usrs.cpp@36@N@foo@F@bar#I#@z Extent=[3:12 - 3:17]
// CHECK: usrs.cpp c:@N@bar Extent=[5:1 - 8:2]
// CHECK: usrs.cpp c:usrs.cpp@N@bar@T@QType Extent=[6:3 - 6:20]
// CHECK: usrs.cpp c:@N@bar@F@bar#I# Extent=[7:3 - 7:20]
// CHECK: usrs.cpp c:usrs.cpp@94@N@bar@F@bar#I#@z Extent=[7:12 - 7:19]
// CHECK: usrs.cpp c:@S@ClsA Extent=[10:1 - 14:2]
// CHECK: usrs.cpp c: Extent=[11:1 - 11:8]
// CHECK: usrs.cpp c:@S@ClsA@FI@a Extent=[12:3 - 12:8]
// CHECK: usrs.cpp c:@S@ClsA@FI@b Extent=[12:3 - 12:11]
// CHECK: usrs.cpp c:@S@ClsA@F@ClsA#I#I# Extent=[13:3 - 13:37]
// CHECK: usrs.cpp c:usrs.cpp@147@S@ClsA@F@ClsA#I#I#@A Extent=[13:8 - 13:13]
// CHECK: usrs.cpp c:usrs.cpp@154@S@ClsA@F@ClsA#I#I#@B Extent=[13:15 - 13:20]
// CHECK: usrs.cpp c:@N@foo Extent=[16:1 - 22:2]
// CHECK: usrs.cpp c:@N@foo@S@ClsB Extent=[17:3 - 21:4]
// CHECK: usrs.cpp c: Extent=[18:3 - 18:10]
// CHECK: usrs.cpp c:@N@foo@S@ClsB@F@ClsB# Extent=[19:5 - 19:27]
// CHECK: usrs.cpp c:@N@foo@S@ClsB@F@result#1 Extent=[20:5 - 20:23]
// CHECK: usrs.cpp c:@N@foo@S@ClsB@F@result#1 Extent=[24:1 - 26:2]
// CHECK: usrs.cpp c:usrs.cpp@aN@S@ClsC Extent=[29:3 - 29:35]
// CHECK: usrs.cpp c:usrs.cpp@aN@w Extent=[30:3 - 30:8]
// CHECK: usrs.cpp c:@z Extent=[33:1 - 33:6]
// CHECK: usrs.cpp c:@N@foo Extent=[35:1 - 40:2]
// CHECK: usrs.cpp c:@N@foo@N@taz Extent=[35:17 - 39:2]
// CHECK: usrs.cpp c:@N@foo@N@taz@x Extent=[36:3 - 36:8]
// CHECK: usrs.cpp c:usrs.cpp@N@foo@N@taz@F@add#I#I# Extent=[37:3 - 37:56]
// CHECK: usrs.cpp c:usrs.cpp@479@N@foo@N@taz@F@add#I#I#@a Extent=[37:25 - 37:30]
// CHECK: usrs.cpp c:usrs.cpp@486@N@foo@N@taz@F@add#I#I#@b Extent=[37:32 - 37:37]
// CHECK: usrs.cpp c:@N@foo@N@taz@F@sub#I#I# Extent=[38:3 - 38:25]
// CHECK: usrs.cpp c:usrs.cpp@522@N@foo@N@taz@F@sub#I#I#@a Extent=[38:12 - 38:17]
// CHECK: usrs.cpp c:usrs.cpp@529@N@foo@N@taz@F@sub#I#I#@b Extent=[38:19 - 38:24]
// CHECK: usrs.cpp c:@N@foo Extent=[42:1 - 52:3]
// CHECK: usrs.cpp c:@N@foo@N@taz Extent=[42:17 - 52:2]
// CHECK: usrs.cpp c:@N@foo@N@taz@S@ClsD Extent=[43:3 - 51:4]
// CHECK: usrs.cpp c: Extent=[44:3 - 44:10]
// CHECK: usrs.cpp c:@N@foo@N@taz@S@ClsD@F@operator=#I# Extent=[45:5 - 45:52]
// CHECK: usrs.cpp c:usrs.cpp@638@N@foo@N@taz@S@ClsD@F@operator=#I#@x Extent=[45:21 - 45:26]
// CHECK: usrs.cpp c:@N@foo@N@taz@S@ClsD@F@operator=#d# Extent=[46:5 - 46:61]
// CHECK: usrs.cpp c:usrs.cpp@690@N@foo@N@taz@S@ClsD@F@operator=#d#@x Extent=[46:21 - 46:29]
// CHECK: usrs.cpp c:@N@foo@N@taz@S@ClsD@F@operator=#&1$@N@foo@N@taz@S@ClsD# Extent=[47:5 - 47:62]
// CHECK: usrs.cpp c:usrs.cpp@751@N@foo@N@taz@S@ClsD@F@operator=#&1$@N@foo@N@taz@S@ClsD#@x Extent=[47:21 - 47:34]
// CHECK: usrs.cpp c:@N@foo@N@taz@S@ClsD@F@qux#S Extent=[48:5 - 48:21]
// CHECK: usrs.cpp c:@N@foo@N@taz@S@ClsD@F@uz#I.#S Extent=[49:5 - 49:30]
// CHECK: usrs.cpp c:usrs.cpp@833@N@foo@N@taz@S@ClsD@F@uz#I.#S@z Extent=[49:19 - 49:24]
// CHECK: usrs.cpp c:@N@foo@N@taz@S@ClsD@F@operator==#&1$@N@foo@N@taz@S@ClsD#1 Extent=[50:5 - 50:62]
// CHECK: usrs.cpp c:usrs.cpp@866@N@foo@N@taz@S@ClsD@F@operator==#&1$@N@foo@N@taz@S@ClsD#1@x Extent=[50:21 - 50:34]
// CHECK: usrs.cpp c:@F@rez Extent=[55:3 - 55:25]
// CHECK: usrs.cpp c:usrs.cpp@941@F@rez@a Extent=[55:12 - 55:17]
// CHECK: usrs.cpp c:usrs.cpp@948@F@rez@b Extent=[55:19 - 55:24]
// CHECK: usrs.cpp c:@NA@foo_alias
// CHECK-NOT: foo
// CHECK: usrs.cpp c:@NA@foo_alias2
// CHECK: usrs.cpp c:usrs.cpp@UD@N@foo@ClsB Extent=[64:1 - 64:16]
// CHECK: usrs.cpp c:@NA@foo_alias3
// CHECK: usrs.cpp c:@aN Extent=[68:1 - 73:2]
// CHECK: usrs.cpp c:usrs.cpp@aN@S@RDar9371763_Foo Extent=[69:1 - 72:2]
// CHECK: usrs.cpp c: Extent=[70:1 - 70:8]
// CHECK: usrs.cpp c:usrs.cpp@aN@S@RDar9371763_Foo@F@bar# Extent=[71:3 - 71:13]
// CHECK: usrs.cpp c:usrs.cpp@aN@S@RDar9371763_Foo@F@bar# Extent=[75:1 - 75:31]
// CHECK: usrs.cpp c:@F@rdar9371763# Extent=[77:1 - 80:2]
// CHECK: usrs.cpp c:usrs.cpp@1204@F@rdar9371763#@foo Extent=[78:3 - 78:22]

// CHECK: usrs.cpp c:@FT@>1#TsomeTemplFn#^type-parameter-0-0:::A# Extent=[82:1 - 82:50]
// CHECK: usrs.cpp c:@FT@>1#TsomeTemplFn#^type-parameter-0-0:::B# Extent=[83:1 - 83:50]
// CHECK: usrs.cpp c:@FT@>1#TsomeTemplFn#I# Extent=[84:1 - 84:40]

// CHECK: usrs.cpp c:@F@funWithChar#C# Extent=[86:1 - 86:28]
// CHECK: usrs.cpp c:@F@funWithChar#c# Extent=[87:1 - 87:37]
// CHECK: usrs.cpp c:@F@funWithChar#r# Extent=[88:1 - 88:35]

// CHECK: usrs.cpp c:usrs.cpp@S@usrs.cpp@1483@FI@x Extent=[90:10 - 90:15]
// CHECK: usrs.cpp c:usrs.cpp@S@usrs.cpp@1510@FI@x Extent=[91:10 - 91:15]

// CHECK: usrs.cpp c:@ST>1#T@TC1@F@meth#>@ST>1#T@TC11t0.0# Extent=[95:3 - 95:17]

// CHECK: usrs.cpp c:@F@vectorOverload#]3f# Extent=[102:1 - 102:32]
// CHECK: usrs.cpp c:@F@vectorOverload#[4f# Extent=[104:1 - 104:32]
