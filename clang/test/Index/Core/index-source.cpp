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
