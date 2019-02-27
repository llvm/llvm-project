// from SemaCXX/class-layout.cpp
// RUN: c-index-test -test-print-type-size %s -target x86_64-pc-linux-gnu | FileCheck -check-prefix=CHECK64 %s
// RUN: c-index-test -test-print-type-size %s -target i386-apple-darwin9 | FileCheck -check-prefix=CHECK32 %s

namespace basic {

// CHECK64: VarDecl=v:[[@LINE+2]]:6 (Definition) (invalid) [type=void] [typekind=Void]
// CHECK32: VarDecl=v:[[@LINE+1]]:6 (Definition) (invalid) [type=void] [typekind=Void]
void v;

// CHECK64: VarDecl=v1:[[@LINE+2]]:7 (Definition) [type=void *] [typekind=Pointer] [sizeof=8] [alignof=8]
// CHECK32: VarDecl=v1:[[@LINE+1]]:7 (Definition) [type=void *] [typekind=Pointer] [sizeof=4] [alignof=4]
void *v1;

// offsetof
// CHECK64: StructDecl=simple:[[@LINE+2]]:8 (Definition) [type=basic::simple] [typekind=Record] [sizeof=48] [alignof=8]
// CHECK32: StructDecl=simple:[[@LINE+1]]:8 (Definition) [type=basic::simple] [typekind=Record] [sizeof=36] [alignof=4]
struct simple {
  int a;
  char b;
// CHECK64: FieldDecl=c:[[@LINE+1]]:7 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=40] [BitFieldSize=3]
  int c:3;
  long d;
  int e:5;
// CHECK64: FieldDecl=f:[[@LINE+1]]:7 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=133] [BitFieldSize=4]
  int f:4;
// CHECK64: FieldDecl=g:[[@LINE+2]]:13 (Definition) [type=long long] [typekind=LongLong] [sizeof=8] [alignof=8] [offsetof=192]
// CHECK32: FieldDecl=g:[[@LINE+1]]:13 (Definition) [type=long long] [typekind=LongLong] [sizeof=8] [alignof=4] [offsetof=128]
  long long g;
// CHECK64: FieldDecl=h:[[@LINE+1]]:8 (Definition) [type=char] [typekind=Char_S] [sizeof=1] [alignof=1] [offsetof=256] [BitFieldSize=3]
  char h:3;
  char i:3;
  float j;
// CHECK64: FieldDecl=k:[[@LINE+2]]:10 (Definition) [type=char *] [typekind=Pointer] [sizeof=8] [alignof=8] [offsetof=320]
// CHECK32: FieldDecl=k:[[@LINE+1]]:10 (Definition) [type=char *] [typekind=Pointer] [sizeof=4] [alignof=4] [offsetof=256]
  char * k;
};


// CHECK64: UnionDecl=u:[[@LINE+2]]:7 (Definition) [type=basic::u] [typekind=Record] [sizeof=48] [alignof=8]
// CHECK32: UnionDecl=u:[[@LINE+1]]:7 (Definition) [type=basic::u] [typekind=Record] [sizeof=36] [alignof=4]
union u {
  int u1;
  long long u2;
  struct simple s1;
};

// CHECK64: VarDecl=s1:[[@LINE+2]]:8 (Definition) [type=basic::simple] [typekind=Record] [sizeof=48] [alignof=8]
// CHECK32: VarDecl=s1:[[@LINE+1]]:8 (Definition) [type=basic::simple] [typekind=Record] [sizeof=36] [alignof=4]
simple s1;

struct Test {
  struct {
    union {
//CHECK64: FieldDecl=foo:[[@LINE+1]]:11 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=0]
      int foo;
    };
  };
};

struct Test2 {
  struct {
    struct {
//CHECK64: FieldDecl=foo:[[@LINE+1]]:11 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=0]
      int foo;
    };
    struct {
//CHECK64: FieldDecl=bar:[[@LINE+1]]:11 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=32/0]
      int bar;
    };
    struct {
        struct {
//CHECK64: FieldDecl=foobar:[[@LINE+1]]:15 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=64/0]
          int foobar;
        };
    };
  };
};

}

// these are test crash. Offsetof return values are not important.
namespace Incomplete {
// test that fields in incomplete named record do not crash
union named {
  struct forward_decl f1;
  int f2;
  struct x {
    int g1;
  } f3;
  struct forward_decl f4;
  struct x2{
    int g2;
    struct forward_decl g3;
  } f5;
};

// test that fields in incomplete anonymous record do not crash
union f {
  struct forward_decl f1;
  int f2;
  struct {
    int e1;
    struct {
      struct forward_decl2 g1;
    };
    int e3;
  };
};


// incomplete not in root level, in named record
struct s1 {
  struct {
    struct forward_decl2 s1_g1;
    int s1_e1;
  } s1_x; // named record shows in s1->field_iterator
  int s1_e3;
};

// incomplete not in root level, in anonymous record
struct s1b {
  struct {
    struct forward_decl2 s1b_g1;
  }; // erroneous anonymous record does not show in s1b->field_iterator
  int s1b_e2;
};

struct s2 {
  struct {
    struct forward_decl2 s2_g1;
    int s2_e1;
  }; // erroneous anonymous record does not show in s1b->field_iterator
  int s2_e3;
};

//deep anonymous with deep level incomplete
struct s3 {
  struct {
    int s3_e1;
    struct {
      struct {
        struct {
          struct {
           struct forward_decl2 s3_g1;
          };
        };
      };
    };
    int s3_e3;
  };
};

//deep anonymous with first level incomplete
struct s4a {
  struct forward_decl2 g1;
  struct {
   struct forward_decl2 g2;
    struct {
      struct {
        struct {
          struct {
//CHECK64: FieldDecl=s4_e1:[[@LINE+1]]:17 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=-1/0]
            int s4_e1;
          };
        };
      };
    };
    int s4_e3;
  };
};

//deep anonymous with sub-first-level incomplete
struct s4b {
  struct {
    struct forward_decl2 g1;
    struct {
      struct {
        struct {
          struct {
            int s4b_e1;
          };
        };
      };
    };
    int s4b_e3;
  };
};

//named struct within anonymous struct
struct s5 {
  struct {
    struct x {
      int i;
    };
  };
};

// CHECK64: StructDecl=As:[[@LINE+1]]:8 [type=Incomplete::As] [typekind=Record]
struct As;

// undefined class. Should not crash
// CHECK64: ClassDecl=A:[[@LINE+1]]:7 [type=Incomplete::A] [typekind=Record]
class A;
class B {
  A* a1;
  A& a2;
};

}

namespace Sizes {

// CHECK64: StructDecl=A:[[@LINE+2]]:8 (Definition) [type=Sizes::A] [typekind=Record] [sizeof=8] [alignof=4]
// CHECK32: StructDecl=A:[[@LINE+1]]:8 (Definition) [type=Sizes::A] [typekind=Record] [sizeof=8] [alignof=4]
struct A {
  int a;
  char b;
};

// CHECK64: StructDecl=B:[[@LINE+2]]:8 (Definition) [type=Sizes::B] [typekind=Record] [sizeof=12] [alignof=4]
// CHECK32: StructDecl=B:[[@LINE+1]]:8 (Definition) [type=Sizes::B] [typekind=Record] [sizeof=12] [alignof=4]
struct B : A {
  char c;
};

// CHECK64: StructDecl=C:[[@LINE+2]]:8 (Definition) [type=Sizes::C] [typekind=Record] [sizeof=8] [alignof=4]
// CHECK32: StructDecl=C:[[@LINE+1]]:8 (Definition) [type=Sizes::C] [typekind=Record] [sizeof=8] [alignof=4]
struct C {
// Make fields private so C won't be a POD type.
private:
  int a;
  char b;
};

// CHECK64: StructDecl=D:[[@LINE+2]]:8 (Definition) [type=Sizes::D] [typekind=Record] [sizeof=8] [alignof=4]
// CHECK32: StructDecl=D:[[@LINE+1]]:8 (Definition) [type=Sizes::D] [typekind=Record] [sizeof=8] [alignof=4]
struct D : C {
  char c;
};

// CHECK64: StructDecl=E:[[@LINE+2]]:32 (Definition) [type=Sizes::E] [typekind=Record] [sizeof=5] [alignof=1]
// CHECK32: StructDecl=E:[[@LINE+1]]:32 (Definition) [type=Sizes::E] [typekind=Record] [sizeof=5] [alignof=1]
struct __attribute__((packed)) E {
  char b;
  int a;
};

// CHECK64: StructDecl=F:[[@LINE+2]]:32 (Definition) [type=Sizes::F] [typekind=Record] [sizeof=6] [alignof=1]
// CHECK32: StructDecl=F:[[@LINE+1]]:32 (Definition) [type=Sizes::F] [typekind=Record] [sizeof=6] [alignof=1]
struct __attribute__((packed)) F : E {
  char d;
};

struct G { G(); };
// CHECK64: StructDecl=H:[[@LINE+2]]:8 (Definition) [type=Sizes::H] [typekind=Record] [sizeof=1] [alignof=1]
// CHECK32: StructDecl=H:[[@LINE+1]]:8 (Definition) [type=Sizes::H] [typekind=Record] [sizeof=1] [alignof=1]
struct H : G { };

// CHECK64: StructDecl=I:[[@LINE+2]]:8 (Definition) [type=Sizes::I] [typekind=Record] [sizeof=5] [alignof=1]
// CHECK32: StructDecl=I:[[@LINE+1]]:8 (Definition) [type=Sizes::I] [typekind=Record] [sizeof=5] [alignof=1]
struct I {
  char b;
  int a;
} __attribute__((packed));

}

namespace Test1 {

// Test complex class hierarchy
struct A { };
struct B : A { virtual void b(); };
class C : virtual A { int c; };
struct D : virtual B { };
struct E : C, virtual D { };
class F : virtual E { };
// CHECK64: StructDecl=G:[[@LINE+2]]:8 (Definition) [type=Test1::G] [typekind=Record] [sizeof=24] [alignof=8]
// CHECK32: StructDecl=G:[[@LINE+1]]:8 (Definition) [type=Test1::G] [typekind=Record] [sizeof=16] [alignof=4]
struct G : virtual E, F { };

}

namespace Test2 {

// Test that this somewhat complex class structure is laid out correctly.
struct A { };
struct B : A { virtual void b(); };
struct C : virtual B { };
struct D : virtual A { };
struct E : virtual B, D { };
struct F : E, virtual C { };
struct G : virtual F, A { };
// CHECK64: StructDecl=H:[[@LINE+2]]:8 (Definition) [type=Test2::H] [typekind=Record] [sizeof=24] [alignof=8]
// CHECK32: StructDecl=H:[[@LINE+1]]:8 (Definition) [type=Test2::H] [typekind=Record] [sizeof=12] [alignof=4]
struct H { G g; };

}

namespace Test3 {
// CHECK64: ClassDecl=B:[[@LINE+2]]:7 (Definition) [type=Test3::B] [typekind=Record] [sizeof=16] [alignof=8]
// CHECK32: ClassDecl=B:[[@LINE+1]]:7 (Definition) [type=Test3::B] [typekind=Record] [sizeof=8] [alignof=4]
class B {
public:
  virtual void b(){}
// CHECK64: FieldDecl=b_field:[[@LINE+2]]:8 (Definition) [type=long] [typekind=Long] [sizeof=8] [alignof=8] [offsetof=64]
// CHECK32: FieldDecl=b_field:[[@LINE+1]]:8 (Definition) [type=long] [typekind=Long] [sizeof=4] [alignof=4] [offsetof=32]
  long b_field;
protected:
private:
};

// CHECK32: ClassDecl=A:[[@LINE+1]]:7 (Definition) [type=Test3::A] [typekind=Record] [sizeof=16] [alignof=4]
class A : public B {
public:
// CHECK64: FieldDecl=a_field:[[@LINE+2]]:7 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=128]
// CHECK32: FieldDecl=a_field:[[@LINE+1]]:7 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=64]
  int a_field;
  virtual void a(){}
// CHECK64: FieldDecl=one:[[@LINE+2]]:8 (Definition) [type=char] [typekind=Char_S] [sizeof=1] [alignof=1] [offsetof=160]
// CHECK32: FieldDecl=one:[[@LINE+1]]:8 (Definition) [type=char] [typekind=Char_S] [sizeof=1] [alignof=1] [offsetof=96]
  char one;
protected:
private:
};

// CHECK64: ClassDecl=D:[[@LINE+2]]:7 (Definition) [type=Test3::D] [typekind=Record] [sizeof=16] [alignof=8]
// CHECK32: ClassDecl=D:[[@LINE+1]]:7 (Definition) [type=Test3::D] [typekind=Record] [sizeof=12] [alignof=4]
class D {
public:
  virtual void b(){}
// CHECK64: FieldDecl=a:[[@LINE+2]]:10 (Definition) [type=double] [typekind=Double] [sizeof=8] [alignof=8] [offsetof=64]
// CHECK32: FieldDecl=a:[[@LINE+1]]:10 (Definition) [type=double] [typekind=Double] [sizeof=8] [alignof=4] [offsetof=32]
  double a;
};

// CHECK64: ClassDecl=C:[[@LINE+2]]:7 (Definition) [type=Test3::C] [typekind=Record] [sizeof=88] [alignof=8]
// CHECK32: ClassDecl=C:[[@LINE+1]]:7 (Definition) [type=Test3::C] [typekind=Record] [sizeof=60] [alignof=4]
class C : public virtual A,
          public D, public B {
public:
  double c1_field;
  int c2_field;
  double c3_field;
  int c4_field;
  virtual void foo(){}
  virtual void bar(){}
protected:
private:
};

struct BaseStruct
{
    BaseStruct(){}
    double v0;
    float v1;
// CHECK64: FieldDecl=fg:[[@LINE+2]]:7 (Definition) [type=Test3::C] [typekind=Record] [sizeof=88] [alignof=8] [offsetof=128]
// CHECK32: FieldDecl=fg:[[@LINE+1]]:7 (Definition) [type=Test3::C] [typekind=Record] [sizeof=60] [alignof=4] [offsetof=96]
    C fg;
// CHECK64: FieldDecl=rg:[[@LINE+2]]:8 (Definition) [type=Test3::C &] [typekind=LValueReference] [sizeof=88] [alignof=8] [offsetof=832]
// CHECK32: FieldDecl=rg:[[@LINE+1]]:8 (Definition) [type=Test3::C &] [typekind=LValueReference] [sizeof=60] [alignof=4] [offsetof=576]
    C &rg;
    int x;
};

}

namespace NotConstantSize {

void f(int i) {
// CHECK32: VarDecl=v2:[[@LINE+1]]:8 (Definition) [type=int [i]] [typekind=VariableArray] [sizeof=-4] [alignof=4]
   int v2[i];
   {
   struct CS1 {
     int f1[i];
     float f2;
   };
   }
}

}

namespace CrashTest {
// test crash scenarios on dependent types.
template<typename T>
struct Foo {
  T t;
  int a;
};

Foo<Sizes::A> t1;
Foo<Sizes::I> t2;

void c;

plopplop;

// CHECK64: StructDecl=lastValid:[[@LINE+2]]:8 (Definition) [type=CrashTest::lastValid] [typekind=Record] [sizeof=1] [alignof=1]
// CHECK32: StructDecl=lastValid:[[@LINE+1]]:8 (Definition) [type=CrashTest::lastValid] [typekind=Record] [sizeof=1] [alignof=1]
struct lastValid {
};

// CHECK64: CXXMethod=Tie:[[@LINE+3]]:8 (const) [type=auto (void *) const] [typekind=FunctionProto] [sizeof=1] [alignof=4] [resulttype=auto] [resulttypekind=Auto] [resultsizeof=-6] [resultalignof=-6]
// CHECK32: CXXMethod=Tie:[[@LINE+2]]:8 (const) [type=auto (void *) const] [typekind=FunctionProto] [sizeof=1] [alignof=4] [resulttype=auto] [resulttypekind=Auto] [resultsizeof=-6] [resultalignof=-6]
class BrowsingContext {
  auto Tie(void*) const;
};

}
