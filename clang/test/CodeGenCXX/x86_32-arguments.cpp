// RUN: %clang_cc1 -triple i386-apple-darwin9 -emit-llvm -o - %s | FileCheck %s

// Non-trivial dtors, should both be passed indirectly.
struct S {
  ~S();
  short s;
};

// CHECK-LABEL: define void @_Z1fv(%struct.S* noalias sret %
S f() { return S(); }
// CHECK-LABEL: define void @_Z1f1S(%struct.S*)
void f(S) { }

// Non-trivial dtors, should both be passed indirectly.
class C {
public:
  ~C();
  double c;
};

// CHECK-LABEL: define void @_Z1gv(%class.C* noalias sret %
C g() { return C(); }

// CHECK-LABEL: define void @_Z1f1C(%class.C*) 
void f(C) { }




// PR7058 - Missing byval on MI thunk definition.

// CHECK-LABEL: define void @_ZThn4_N18BasicAliasAnalysis13getModRefInfoE8CallSite
// ...
// CHECK: %struct.CallSite* byval(%struct.CallSite) align 4 %CS)
struct CallSite {
  unsigned Ptr;
  CallSite(unsigned XX) : Ptr(XX) {}
};

struct AliasAnalysis {
  virtual void xyz();
  virtual void getModRefInfo(CallSite CS) = 0;
};

struct ModulePass {
  virtual void xx();
};

struct BasicAliasAnalysis : public ModulePass, public AliasAnalysis {
  void getModRefInfo(CallSite CS);
};

void BasicAliasAnalysis::getModRefInfo(CallSite CS) {
}

// Check various single element struct type conditions.
//
// PR7098.

// CHECK-LABEL: define i64 @_Z2f0v()
struct s0_0 { int x; };
struct s0_1 : s0_0 { int* y; };
s0_1 f0() { return s0_1(); }

// CHECK-LABEL: define i32 @_Z2f1v()
struct s1_0 { int x; };
struct s1_1 : s1_0 { };
s1_1 f1() { return s1_1(); }

// CHECK-LABEL: define double @_Z2f2v()
struct s2_0 { double x; };
struct s2_1 : s2_0 { };
s2_1 f2() { return s2_1(); }

// CHECK-LABEL: define double @_Z2f3v()
struct s3_0 { };
struct s3_1 { double x; };
struct s3_2 : s3_0, s3_1 { };
s3_2 f3() { return s3_2(); }

// CHECK-LABEL: define i64 @_Z2f4v()
struct s4_0 { float x; };
struct s4_1 { float x; };
struct s4_2 : s4_0, s4_1 { };
s4_2 f4() { return s4_2(); }

// CHECK-LABEL: define i32* @_Z2f5v()
struct s5 { s5(); int &x; };
s5 f5() { return s5(); }

// CHECK-LABEL: define i32 @_Z4f6_0M2s6i(i32 %a)
// CHECK: define i64 @_Z4f6_1M2s6FivE({ i32, i32 }* byval({ i32, i32 }) align 4)
// FIXME: It would be nice to avoid byval on the previous case.
struct s6 {};
typedef int s6::* s6_mdp;
typedef int (s6::*s6_mfp)();
s6_mdp f6_0(s6_mdp a) { return a; }
s6_mfp f6_1(s6_mfp a) { return a; }

// CHECK-LABEL: define double @_Z2f7v()
struct s7_0 { unsigned : 0; };
struct s7_1 { double x; };
struct s7 : s7_0, s7_1 { };
s7 f7() { return s7(); }

// CHECK-LABEL: define void @_Z2f8v(%struct.s8* noalias sret %agg.result)
struct s8_0 { };
struct s8_1 { double x; };
struct s8 { s8_0 a; s8_1 b; };
s8 f8() { return s8(); }

// CHECK-LABEL: define void @_Z2f9v(%struct.s9* noalias sret %agg.result)
struct s9_0 { unsigned : 0; };
struct s9_1 { double x; };
struct s9 { s9_0 a; s9_1 b; };
s9 f9() { return s9(); }
