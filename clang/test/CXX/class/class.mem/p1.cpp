// RUN: %clang_cc1 -fsyntax-only -verify %s 

struct S 
{
  static int v1; // expected-note{{previous declaration is here}}
  int v1; //expected-error{{duplicate member 'v1'}}
  int v;  //expected-note 2{{previous definition is here}} \
          // expected-note{{previous declaration is here}}
  static int v; //expected-error{{redefinition of 'v' as different kind of symbol}}
  int v; //expected-error{{duplicate member 'v'}}
  static int v; //expected-error{{redefinition of 'v' as different kind of symbol}}
  enum EnumT { E = 10 }; // expected-note {{declared here}}
  friend struct M;
  struct X;  //expected-note{{forward declaration of 'S::X'}}
  friend struct X;
};

S::EnumT Evar = S::E; // ok
S::EnumT Evar2 = EnumT(); //expected-error{{use of undeclared identifier 'EnumT'; did you mean 'S::EnumT'?}}
S::M m; //expected-error{{no type named 'M' in 'S'}}
S::X x; //expected-error{{variable has incomplete type 'S::X'}}


struct S2 
{
  static int v2; // expected-note{{previous declaration is here}}
  static int v2; //expected-error{{duplicate member 'v2'}}
};

struct S3
{
  static int v3;
  struct S4
  {
    static int v3;
  };
};

struct S4
{
  static int v4;
};

int S4::v4; //expected-note{{previous definition is here}}
int S4::v4; //expected-error{{redefinition of 'v4'}}

struct S5
{
  static int v5; //expected-note{{previous definition is here}}
  void v5() { } //expected-error{{redefinition of 'v5' as different kind of symbol}}
  
  void v6() { } //expected-note{{previous definition is here}}
  static int v6; //expected-error{{redefinition of 'v6' as different kind of symbol}}
  
  void v7() { }
  void v7(int) { } //expected-note{{previous definition is here}}
  static int v7;  //expected-error{{redefinition of 'v7' as different kind of symbol}}
  
  void v8();
  int v8(int); //expected-note{{previous declaration is here}}
  int v8; //expected-error{{duplicate member 'v8'}}
  
  
};

namespace PR8245 {
  class X {
  public:
    template<class C> 
    class Inner {
    public:
      void foo(bool bar = true);
      int bam;
    };

    Inner<int> _foo;    
  };

  void f() {
    X::Inner<int> c2i;
    X::Inner<float> c2f;
    c2i.foo();
    c2f.foo();
  }

  class Y {
    class Inner {
      void foo(int = sizeof(Y));
    };
  };
}
