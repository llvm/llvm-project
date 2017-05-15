// RUN: c-index-test core -print-source-symbols -- %s -std=c++14 -target x86_64-apple-macosx10.7 | FileCheck %s

// CHECK: [[@LINE+1]]:7 | class/C++ | Cls | [[Cls_USR:.*]] | <no-cgname> | Def | rel: 0
class Cls {
  // CHECK: [[@LINE+3]]:3 | constructor/C++ | Cls | c:@S@Cls@F@Cls#I# | __ZN3ClsC1Ei | Decl,RelChild | rel: 1
  // CHECK-NEXT: RelChild | Cls | c:@S@Cls
  // CHECK: [[@LINE+1]]:3 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref,RelCont | rel: 1
  Cls(int x);
  // CHECK: [[@LINE+2]]:3 | constructor/cxx-copy-ctor/C++ | Cls | c:@S@Cls@F@Cls#&1$@S@Cls# | __ZN3ClsC1ERKS_ | Decl,RelChild | rel: 1
  // CHECK: [[@LINE+1]]:3 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref,RelCont | rel: 1
  Cls(const Cls &);
  // CHECK: [[@LINE+2]]:3 | constructor/cxx-move-ctor/C++ | Cls | c:@S@Cls@F@Cls#&&$@S@Cls# | __ZN3ClsC1EOS_ | Decl,RelChild | rel: 1
  // CHECK: [[@LINE+1]]:3 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref,RelCont | rel: 1
  Cls(Cls &&);

  // CHECK: [[@LINE+2]]:3 | destructor/C++ | ~Cls | c:@S@Cls@F@~Cls# | __ZN3ClsD1Ev | Decl,RelChild | rel: 1
  // CHECK: [[@LINE+1]]:4 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref,RelCont | rel: 1
  ~Cls();
};

// CHECK: [[@LINE+3]]:7 | class/C++ | SubCls1 | [[SubCls1_USR:.*]] | <no-cgname> | Def | rel: 0
// CHECK: [[@LINE+2]]:24 | class/C++ | Cls | [[Cls_USR]] | <no-cgname> | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | SubCls1 | [[SubCls1_USR]]
class SubCls1 : public Cls {};
// CHECK: [[@LINE+1]]:13 | type-alias/C | ClsAlias | [[ClsAlias_USR:.*]] | <no-cgname> | Def | rel: 0
typedef Cls ClsAlias;
// CHECK: [[@LINE+5]]:7 | class/C++ | SubCls2 | [[SubCls2_USR:.*]] | <no-cgname> | Def | rel: 0
// CHECK: [[@LINE+4]]:24 | type-alias/C | ClsAlias | [[ClsAlias_USR]] | <no-cgname> | Ref,RelCont | rel: 1
// CHECK-NEXT: RelCont | SubCls2 | [[SubCls2_USR]]
// CHECK: [[@LINE+2]]:24 | class/C++ | Cls | [[Cls_USR]] | <no-cgname> | Ref,Impl,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | SubCls2 | [[SubCls2_USR]]
class SubCls2 : public ClsAlias {};

Cls::Cls(int x) {}
// CHECK: [[@LINE-1]]:6 | constructor/C++ | Cls | c:@S@Cls@F@Cls#I# | __ZN3ClsC1Ei | Def,RelChild | rel: 1
// CHECK: [[@LINE-2]]:1 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref,RelCont | rel: 1
// CHECK: [[@LINE-3]]:6 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref,RelCont | rel: 1

Cls::~/*a comment*/Cls() {}
// CHECK: [[@LINE-1]]:6 | destructor/C++ | ~Cls | c:@S@Cls@F@~Cls# | __ZN3ClsD1Ev | Def,RelChild | rel: 1
// CHECK: [[@LINE-2]]:1 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref,RelCont | rel: 1
// CHECK: [[@LINE-3]]:20 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref,RelCont | rel: 1

template <typename TemplArg>
class TemplCls {
// CHECK: [[@LINE-1]]:7 | class(Gen)/C++ | TemplCls | c:@ST>1#T@TemplCls | <no-cgname> | Def | rel: 0
public:
  TemplCls(int x);
  // CHECK: [[@LINE-1]]:3 | constructor/C++ | TemplCls | c:@ST>1#T@TemplCls@F@TemplCls#I# | <no-cgname> | Decl,RelChild | rel: 1
  // CHECK-NEXT: RelChild | TemplCls | c:@ST>1#T@TemplCls
};

template<>
class TemplCls<double> {
// CHECK: [[@LINE-1]]:7 | class(Gen,TS)/C++ | TemplCls | c:@S@TemplCls>#d | <no-cgname> | Def,RelSpecialization | rel: 1
// CHECK: RelSpecialization | TemplCls | c:@ST>1#T@TemplCls
};

TemplCls<int> gtv(0);
// CHECK: [[@LINE-1]]:1 | class(Gen)/C++ | TemplCls | c:@ST>1#T@TemplCls | <no-cgname> | Ref,RelCont | rel: 1

template<class T>
class Wrapper {};
template<class T, class P>
class Wrapper<T(P)> {};

// CHECK: [[@LINE+1]]:6 | function/C | test1 | [[TEST1_USR:.*]] | [[TEST1_CG:.*]] | Decl | rel: 0
void test1(Wrapper<void(int)> f);
// CHECK: [[@LINE+1]]:6 | function/C | test1 | [[TEST1_USR]] | [[TEST1_CG]] | Def | rel: 0
void test1(Wrapper<void(int)> f) {}

template <typename T>
class BT {
  struct KLR {
    int idx;
  };

  // CHECK: [[@LINE+1]]:7 | instance-method/C++ | foo |
  KLR foo() {
    return { .idx = 0 }; // Make sure this doesn't trigger a crash.
  }
};

// CHECK: [[@LINE+1]]:23 | type-alias/C | size_t |
typedef unsigned long size_t;
// CHECK: [[@LINE+2]]:7 | function/C | operator new | c:@F@operator new#l# | __Znwm |
// CHECK: [[@LINE+1]]:20 | type-alias/C | size_t | {{.*}} | Ref,RelCont |
void* operator new(size_t sz);

// CHECK: [[@LINE+1]]:37 | variable(Gen)/C++ | tmplVar | c:index-source.cpp@VT>1#T@tmplVar | __ZL7tmplVar | Def | rel: 0
template<typename T> static const T tmplVar = T(0);
// CHECK: [[@LINE+1]]:29 | variable(Gen,TS)/C++ | tmplVar | c:index-source.cpp@tmplVar>#I | __ZL7tmplVarIiE | Def | rel: 0
template<> static const int tmplVar<int> = 0;
// CHECK: [[@LINE+2]]:5 | variable/C | gvi | c:@gvi | _gvi | Def | rel: 0
// CHECK: [[@LINE+1]]:11 | variable(Gen,TS)/C++ | tmplVar | c:index-source.cpp@tmplVar>#I | __ZL7tmplVarIiE | Ref,Read,RelCont | rel: 1
int gvi = tmplVar<int>;
// CHECK: [[@LINE+2]]:5 | variable/C | gvf | c:@gvf | _gvf | Def | rel: 0
// CHECK: [[@LINE+1]]:11 | variable(Gen)/C++ | tmplVar | c:index-source.cpp@VT>1#T@tmplVar | __ZL7tmplVar | Ref,Read,RelCont | rel: 1
int gvf = tmplVar<float>;

template<typename A, typename B>
class PartialSpecilizationClass { };
// CHECK: [[@LINE-1]]:7 | class(Gen)/C++ | PartialSpecilizationClass | c:@ST>2#T#T@PartialSpecilizationClass | <no-cgname> | Def | rel: 0

template<typename B>
class PartialSpecilizationClass<int, B *> { };
// CHECK: [[@LINE-1]]:7 | class(Gen,TPS)/C++ | PartialSpecilizationClass | c:@SP>1#T@PartialSpecilizationClass>#I#*t0.0 | <no-cgname> | Def,RelSpecialization | rel: 1
// CHECK-NEXT: RelSpecialization | PartialSpecilizationClass | c:@ST>2#T#T@PartialSpecilizationClass

template<>
class PartialSpecilizationClass<int, int> { };
// CHECK: [[@LINE-1]]:7 | class(Gen,TS)/C++ | PartialSpecilizationClass | c:@S@PartialSpecilizationClass>#I#I | <no-cgname> | Def,RelSpecialization | rel: 1
// CHECK-NEXT: RelSpecialization | PartialSpecilizationClass | c:@ST>2#T#T@PartialSpecilizationClass

template<typename T, typename S>
class PseudoOverridesInSpecializations {
  void function() { }
  void function(int) { }

  static void staticFunction() { }

  int field;
  static int variable;

  typedef T TypeDef;
  using TypeAlias = T;

  enum anEnum { };

  struct Struct { };
  union Union { };

  using TypealiasOrRecord = void;

  template<typename U> struct InnerTemplate { };
  template<typename U> struct InnerTemplate <U*> { };
};

template<>
class PseudoOverridesInSpecializations<double, int> {
  void function() { }
// CHECK: [[@LINE-1]]:8 | instance-method/C++ | function | c:@S@PseudoOverridesInSpecializations>#d#I@F@function# | __ZN32PseudoOverridesInSpecializationsIdiE8functionEv | Def,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | function | c:@ST>2#T#T@PseudoOverridesInSpecializations@F@function#

  void staticFunction() { }
// CHECK: [[@LINE-1]]:8 | instance-method/C++ | staticFunction | c:@S@PseudoOverridesInSpecializations>#d#I@F@staticFunction# | __ZN32PseudoOverridesInSpecializationsIdiE14staticFunctionEv | Def,RelChild | rel: 1
// CHECK-NOT: RelSpecialization

  int notOverridingField = 0;

// CHECK-LABEL: checLabelBreak
  int checLabelBreak = 0;

  int field = 0;
// CHECK: [[@LINE-1]]:7 | field/C++ | field | c:@S@PseudoOverridesInSpecializations>#d#I@FI@field | <no-cgname> | Def,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | field | c:@ST>2#T#T@PseudoOverridesInSpecializations@FI@field

  static double variable;
// CHECK: [[@LINE-1]]:17 | static-property/C++ | variable | c:@S@PseudoOverridesInSpecializations>#d#I@variable | __ZN32PseudoOverridesInSpecializationsIdiE8variableE | Decl,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | variable | c:@ST>2#T#T@PseudoOverridesInSpecializations@variable

  typedef double TypeDef;
// CHECK: [[@LINE-1]]:18 | type-alias/C | TypeDef | c:index-source.cpp@S@PseudoOverridesInSpecializations>#d#I@T@TypeDef | <no-cgname> | Def,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | TypeDef | c:index-source.cpp@ST>2#T#T@PseudoOverridesInSpecializations@T@TypeDef

  using TypeAlias = int;
// CHECK: [[@LINE-1]]:9 | type-alias/C++ | TypeAlias | c:@S@PseudoOverridesInSpecializations>#d#I@TypeAlias | <no-cgname> | Def,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | TypeAlias | c:@ST>2#T#T@PseudoOverridesInSpecializations@TypeAlias

  enum anEnum { };
// CHECK: [[@LINE-1]]:8 | enum/C | anEnum | c:@S@PseudoOverridesInSpecializations>#d#I@E@anEnum | <no-cgname> | Def,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | anEnum | c:@ST>2#T#T@PseudoOverridesInSpecializations@E@anEnum
  class Struct { };
// CHECK: [[@LINE-1]]:9 | class/C++ | Struct | c:@S@PseudoOverridesInSpecializations>#d#I@S@Struct | <no-cgname> | Def,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | Struct | c:@ST>2#T#T@PseudoOverridesInSpecializations@S@Struct
  union Union { };
// CHECK: [[@LINE-1]]:9 | union/C | Union | c:@S@PseudoOverridesInSpecializations>#d#I@U@Union | <no-cgname> | Def,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | Union | c:@ST>2#T#T@PseudoOverridesInSpecializations@U@Union

  struct TypealiasOrRecord { };
// CHECK: [[@LINE-1]]:10 | struct/C | TypealiasOrRecord | c:@S@PseudoOverridesInSpecializations>#d#I@S@TypealiasOrRecord | <no-cgname> | Def,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | TypealiasOrRecord | c:@ST>2#T#T@PseudoOverridesInSpecializations@TypealiasOrRecord

  template<typename U> struct InnerTemplate { };
// CHECK: [[@LINE-1]]:31 | struct(Gen)/C++ | InnerTemplate | c:@S@PseudoOverridesInSpecializations>#d#I@ST>1#T@InnerTemplate | <no-cgname> | Def,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | InnerTemplate | c:@ST>2#T#T@PseudoOverridesInSpecializations@ST>1#T@InnerTemplate
  template<typename U> struct InnerTemplate <U*> { };
};

template<typename S>
class PseudoOverridesInSpecializations<float, S> {
  typedef float TypealiasOrRecord;
// CHECK: [[@LINE-1]]:17 | type-alias/C | TypealiasOrRecord | c:index-source.cpp@SP>1#T@PseudoOverridesInSpecializations>#f#t0.0@T@TypealiasOrRecord | <no-cgname> | Def,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | TypealiasOrRecord | c:@ST>2#T#T@PseudoOverridesInSpecializations@TypealiasOrRecord
};

template<typename T, typename U>
class ConflictingPseudoOverridesInSpecialization {
  void foo(T x);
  void foo(U x);
};

template<typename T>
class ConflictingPseudoOverridesInSpecialization<int, T> {
  void foo(T x);
// CHECK: [[@LINE-1]]:8 | instance-method/C++ | foo | c:@SP>1#T@ConflictingPseudoOverridesInSpecialization>#I#t0.0@F@foo#S0_# | <no-cgname> | Decl,RelChild,RelSpecialization | rel: 3
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | foo | c:@ST>2#T#T@ConflictingPseudoOverridesInSpecialization@F@foo#t0.0#
// CHECK-NEXT: RelSpecialization | foo | c:@ST>2#T#T@ConflictingPseudoOverridesInSpecialization@F@foo#t0.1#
};

template<typename T, typename U, int x>
void functionSpecialization();

template<typename T, typename U, int x>
void functionSpecialization() { }
// CHECK: [[@LINE-1]]:6 | function/C | functionSpecialization | c:@FT@>3#T#T#NIfunctionSpecialization#v# | <no-cgname> | Def | rel: 0

template<>
void functionSpecialization<int, int, 0>();
// CHECK: [[@LINE-1]]:6 | function(Gen,TS)/C++ | functionSpecialization | c:@F@functionSpecialization<#I#I#VI0># | __Z22functionSpecializationIiiLi0EEvv | Decl,RelSpecialization | rel: 1
// CHECK-NEXT: RelSpecialization | functionSpecialization | c:@FT@>3#T#T#NIfunctionSpecialization#v#

template<>
void functionSpecialization<int, int, 0>() { }
// CHECK: [[@LINE-1]]:6 | function(Gen,TS)/C++ | functionSpecialization | c:@F@functionSpecialization<#I#I#VI0># | __Z22functionSpecializationIiiLi0EEvv | Def,RelSpecialization | rel: 1
// CHECK-NEXT: RelSpecialization | functionSpecialization | c:@FT@>3#T#T#NIfunctionSpecialization#v#

struct ContainsSpecializedMemberFunction {
  template<typename T>
  void memberSpecialization();
};

template<typename T>
void ContainsSpecializedMemberFunction::memberSpecialization() {
// CHECK: [[@LINE-1]]:41 | instance-method/C++ | memberSpecialization | c:@S@ContainsSpecializedMemberFunction@FT@>1#TmemberSpecialization#v# | <no-cgname> | Def,RelChild | rel: 1
// CHECK-NEXT: RelChild
}

template<>
void ContainsSpecializedMemberFunction::memberSpecialization<int>() {
// CHECK: [[@LINE-1]]:41 | instance-method(Gen,TS)/C++ | memberSpecialization | c:@S@ContainsSpecializedMemberFunction@F@memberSpecialization<#I># | __ZN33ContainsSpecializedMemberFunction20memberSpecializationIiEEvv | Def,RelChild,RelSpecialization | rel: 2
// CHECK-NEXT: RelChild
// CHECK-NEXT: RelSpecialization | memberSpecialization | c:@S@ContainsSpecializedMemberFunction@FT@>1#TmemberSpecialization#v#
}

template<typename T>
class SpecializationDecl;
// CHECK: [[@LINE-1]]:7 | class(Gen)/C++ | SpecializationDecl | c:@ST>1#T@SpecializationDecl | <no-cgname> | Ref | rel: 0

template<typename T>
class SpecializationDecl { };
// CHECK: [[@LINE-1]]:7 | class(Gen)/C++ | SpecializationDecl | c:@ST>1#T@SpecializationDecl | <no-cgname> | Def | rel: 0

template<>
class SpecializationDecl<int>;
// CHECK: [[@LINE-1]]:7 | class(Gen)/C++ | SpecializationDecl | c:@ST>1#T@SpecializationDecl | <no-cgname> | Ref | rel: 0

template<>
class SpecializationDecl<int> { };
// CHECK: [[@LINE-1]]:7 | class(Gen,TS)/C++ | SpecializationDecl | c:@S@SpecializationDecl>#I | <no-cgname> | Def,RelSpecialization | rel: 1
// CHECK-NEXT: RelSpecialization | SpecializationDecl | c:@ST>1#T@SpecializationDecl
// CHECK-NEXT: [[@LINE-3]]:7 | class(Gen)/C++ | SpecializationDecl | c:@ST>1#T@SpecializationDecl | <no-cgname> | Ref | rel: 0

template<typename T>
class PartialSpecilizationClass<Cls, T>;
// CHECK: [[@LINE-1]]:7 | class(Gen)/C++ | PartialSpecilizationClass | c:@ST>2#T#T@PartialSpecilizationClass | <no-cgname> | Ref | rel: 0
// CHECK-NEXT: [[@LINE-2]]:33 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref | rel: 0

template<>
class PartialSpecilizationClass<Cls, Cls> : Cls { };
// CHECK: [[@LINE-1]]:7 | class(Gen,TS)/C++ | PartialSpecilizationClass | c:@S@PartialSpecilizationClass>#$@S@Cls#S0_ | <no-cgname> | Def,RelSpecialization | rel: 1
// CHECK-NEXT: RelSpecialization | PartialSpecilizationClass | c:@ST>2#T#T@PartialSpecilizationClass
// CHECK-NEXT: [[@LINE-3]]:45 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref,RelBase,RelCont | rel: 1
// CHECK-NEXT: RelBase,RelCont | PartialSpecilizationClass | c:@S@PartialSpecilizationClass>#$@S@Cls#S0_
// CHECK-NEXT: [[@LINE-5]]:7 | class(Gen)/C++ | PartialSpecilizationClass | c:@ST>2#T#T@PartialSpecilizationClass | <no-cgname> | Ref | rel: 0
// CHECK-NEXT: [[@LINE-6]]:33 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref | rel: 0
// CHECK-NEXT: [[@LINE-7]]:38 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref | rel: 0

template<typename T, int x>
void functionSp() { }

struct Record {
  constexpr static int C = 2;
};

template<>
void functionSp<SpecializationDecl<Cls>, Record::C>() {
// CHECK: [[@LINE-1]]:6 | function(Gen,TS)/C++ | functionSp | c:@F@functionSp<#$@S@SpecializationDecl>#$@S@Cls#VI2># | __Z10functionSpI18SpecializationDeclI3ClsELi2EEvv | Def,RelSpecialization | rel: 1
// CHECK:   RelSpecialization | functionSp | c:@FT@>2#T#NIfunctionSp#v#
// CHECK: [[@LINE-3]]:17 | class(Gen)/C++ | SpecializationDecl | c:@ST>1#T@SpecializationDecl | <no-cgname> | Ref,RelCont | rel: 1
// CHECK: [[@LINE-4]]:36 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref,RelCont | rel: 1
// CHECK: [[@LINE-5]]:50 | static-property/C++ | C | c:@S@Record@C | __ZN6Record1CE | Ref,RelCont | rel: 1
// CHECK: [[@LINE-6]]:42 | struct/C++ | Record | c:@S@Record | <no-cgname> | Ref,RelCont | rel: 1
}

template<typename T, int x>
class ClassWithCorrectSpecialization { };

template<>
class ClassWithCorrectSpecialization<SpecializationDecl<Cls>, Record::C> { };
// CHECK: [[@LINE-1]]:38 | class(Gen)/C++ | SpecializationDecl | c:@ST>1#T@SpecializationDecl | <no-cgname> | Ref | rel: 0
// CHECK: [[@LINE-2]]:57 | class/C++ | Cls | c:@S@Cls | <no-cgname> | Ref | rel: 0
// CHECK: [[@LINE-3]]:71 | static-property/C++ | C | c:@S@Record@C | __ZN6Record1CE | Ref,Read | rel: 0
// CHECK: [[@LINE-4]]:63 | struct/C++ | Record | c:@S@Record | <no-cgname> | Ref | rel: 0

namespace ns {
// CHECK: [[@LINE-1]]:11 | namespace/C++ | ns | c:@N@ns | <no-cgname> | Decl | rel: 0
namespace inner {
// CHECK: [[@LINE-1]]:11 | namespace/C++ | inner | c:@N@ns@N@inner | <no-cgname> | Decl,RelChild | rel: 1
void func();

}
namespace innerAlias = inner;
// CHECK: [[@LINE-1]]:11 | namespace-alias/C++ | innerAlias | c:@N@ns@NA@innerAlias | <no-cgname> | Decl,RelChild | rel: 1
// CHECK: [[@LINE-2]]:24 | namespace/C++ | inner | c:@N@ns@N@inner | <no-cgname> | Ref,RelCont | rel: 1
}

namespace namespaceAlias = ::ns::innerAlias;
// CHECK: [[@LINE-1]]:11 | namespace-alias/C++ | namespaceAlias | c:@NA@namespaceAlias | <no-cgname> | Decl | rel: 0
// CHECK: [[@LINE-2]]:30 | namespace/C++ | ns | c:@N@ns | <no-cgname> | Ref,RelCont | rel: 1
// CHECK: [[@LINE-3]]:34 | namespace-alias/C++ | innerAlias | c:@N@ns@NA@innerAlias | <no-cgname> | Ref,RelCont | rel: 1

void ::ns::inner::func() {
// CHECK: [[@LINE-1]]:8 | namespace/C++ | ns | c:@N@ns | <no-cgname> | Ref,RelCont | rel: 1
// CHECK: [[@LINE-2]]:12 | namespace/C++ | inner | c:@N@ns@N@inner | <no-cgname> | Ref,RelCont | rel: 1
  ns::innerAlias::func();
// CHECK: [[@LINE-1]]:3 | namespace/C++ | ns | c:@N@ns | <no-cgname> | Ref,RelCont | rel: 1
// CHECK: [[@LINE-2]]:7 | namespace-alias/C++ | innerAlias | c:@N@ns@NA@innerAlias | <no-cgname> | Ref,RelCont | rel: 1
}
