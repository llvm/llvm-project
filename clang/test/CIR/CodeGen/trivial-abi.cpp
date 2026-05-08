// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// All 17 cases from clang/test/CodeGenCXX/trivial_abi.cpp, adapted for
// x86_64-unknown-linux-gnu with CIR/LLVM/OGCG checks.

struct __attribute__((trivial_abi)) Small {
  int *p;
  Small();
  ~Small();
  Small(const Small &) noexcept;
  Small &operator=(const Small &);
};

struct __attribute__((trivial_abi)) Large {
  int *p;
  int a[128];
  Large();
  ~Large();
  Large(const Large &) noexcept;
  Large &operator=(const Large &);
};

struct Trivial { int a; };

struct NonTrivial {
  NonTrivial();
  ~NonTrivial();
  int a;
};

// --- Case 17: PR42961 lambda returning Small via __invoke ---
// CIR emits lambda internals before user functions, so this CIR check
// must come first to match CIR emission order.

Small (*fp)() = []() -> Small { return Small(); };

// CIR-LABEL: cir.func {{.*}} @{{.*}}__invokeEv
// CIR:   cir.call @{{.*}}clEv
// LLVM-LABEL: define {{.*}} @{{.*}}__invokeEv
// LLVM:   call {{.*}} @{{.*}}clEv

// --- Case 1: D0::m0 thunk returning trivial_abi Small ---

struct B0 { virtual Small m0(); };
struct B1 { virtual Small m0(); };
struct D0 : B0, B1 { Small m0() override; };
Small D0::m0() { return {}; }

// CIR-LABEL: cir.func {{.*}} @_ZThn8_N2D02m0Ev
// CIR:   cir.call @_ZN2D02m0Ev
// LLVM-LABEL: define {{.*}} @_ZN2D02m0Ev(
// LLVM-LABEL: define {{.*}} @_ZThn8_N2D02m0Ev(
// LLVM:   getelementptr i8, ptr {{.*}}, i64 -8
// LLVM:   call {{.*}} @_ZN2D02m0Ev(
// OGCG-LABEL: define {{.*}} @_ZN2D02m0Ev(
// OGCG-LABEL: define {{.*}} @_ZThn8_N2D02m0Ev(
// OGCG:   getelementptr inbounds i8, ptr {{.*}}, i64 -8
// OGCG:   {{.*}}call {{.*}} @_ZN2D02m0Ev(

// --- Case 2: testParamSmall ---

void testParamSmall(Small a) noexcept {}

// CIR-LABEL: cir.func {{.*}} @_Z14testParamSmall5Small
// CIR:   cir.return
// LLVM-LABEL: define {{.*}} void @_Z14testParamSmall5Small(
// LLVM:   ret void
// OGCG-LABEL: define {{.*}} void @_Z14testParamSmall5Small(ptr %a.coerce)
// OGCG:   call {{.*}} @_ZN5SmallD1Ev(
// OGCG:   ret void

// --- Case 3: testReturnSmall ---

Small testReturnSmall() {
  Small t;
  return t;
}

// CIR-LABEL: cir.func {{.*}} @_Z15testReturnSmallv
// CIR:   cir.call @_ZN5SmallC1Ev
// LLVM-LABEL: define {{.*}} @_Z15testReturnSmallv(
// LLVM:   call void @_ZN5SmallC1Ev(
// OGCG-LABEL: define {{.*}} ptr @_Z15testReturnSmallv(
// OGCG:   call {{.*}} @_ZN5SmallC1Ev(

// --- Case 4: testCallSmall0 (local copy + callee-destructed param) ---

void testCallSmall0() {
  Small t;
  testParamSmall(t);
}

// CIR-LABEL: cir.func {{.*}} @_Z14testCallSmall0v
// CIR:   cir.call @_ZN5SmallC1Ev
// CIR:   cir.call @_ZN5SmallC1ERKS_
// CIR:   cir.call @_Z14testParamSmall5Small
// LLVM-LABEL: define {{.*}} void @_Z14testCallSmall0v(
// LLVM:   call void @_ZN5SmallC1Ev(
// LLVM:   call void @_ZN5SmallC1ERKS_(
// LLVM:   call void @_Z14testParamSmall5Small(
// OGCG-LABEL: define {{.*}} void @_Z14testCallSmall0v(
// OGCG:   call {{.*}} @_ZN5SmallC1Ev(
// OGCG:   call {{.*}} @_ZN5SmallC1ERKS_(
// OGCG:   call void @_Z14testParamSmall5Small(

// --- Case 5: testCallSmall1 (pass returned value directly) ---

void testCallSmall1() {
  testParamSmall(testReturnSmall());
}

// CIR-LABEL: cir.func {{.*}} @_Z14testCallSmall1v
// CIR:   cir.call @_Z15testReturnSmallv
// CIR:   cir.call @_Z14testParamSmall5Small
// LLVM-LABEL: define {{.*}} void @_Z14testCallSmall1v(
// LLVM:   call {{.*}} @_Z15testReturnSmallv(
// LLVM:   call void @_Z14testParamSmall5Small(
// OGCG-LABEL: define {{.*}} void @_Z14testCallSmall1v(
// OGCG:   call {{.*}} @_Z15testReturnSmallv()
// OGCG:   call void @_Z14testParamSmall5Small(

// --- Case 6: testIgnoredSmall (discard return, must destruct) ---

void testIgnoredSmall() {
  testReturnSmall();
}

// CIR-LABEL: cir.func {{.*}} @_Z16testIgnoredSmallv
// CIR:   cir.call @_Z15testReturnSmallv
// CIR:   cir.call @_ZN5SmallD1Ev
// LLVM-LABEL: define {{.*}} void @_Z16testIgnoredSmallv(
// LLVM:   call {{.*}} @_Z15testReturnSmallv(
// LLVM:   call void @_ZN5SmallD1Ev(
// OGCG-LABEL: define {{.*}} void @_Z16testIgnoredSmallv(
// OGCG:   call {{.*}} @_Z15testReturnSmallv()
// OGCG:   call {{.*}} @_ZN5SmallD1Ev(

// --- Case 7: testParamLarge ---

void testParamLarge(Large a) noexcept {}

// CIR-LABEL: cir.func {{.*}} @_Z14testParamLarge5Large
// CIR:   cir.return
// LLVM-LABEL: define {{.*}} void @_Z14testParamLarge5Large(
// LLVM:   ret void
// OGCG-LABEL: define {{.*}} void @_Z14testParamLarge5Large(ptr noundef byval(%struct.Large) align 8 %a)
// OGCG:   call {{.*}} @_ZN5LargeD1Ev(
// OGCG:   ret void

// --- Case 8: testReturnLarge ---

Large testReturnLarge() {
  Large t;
  return t;
}

// CIR-LABEL: cir.func {{.*}} @_Z15testReturnLargev
// CIR:   cir.call @_ZN5LargeC1Ev
// LLVM-LABEL: define {{.*}} @_Z15testReturnLargev(
// LLVM:   call void @_ZN5LargeC1Ev(
// OGCG-LABEL: define {{.*}} void @_Z15testReturnLargev(ptr {{.*}}sret(%struct.Large)
// OGCG:   call {{.*}} @_ZN5LargeC1Ev(

// --- Case 9: testCallLarge0 (local copy + callee-destructed param) ---

void testCallLarge0() {
  Large t;
  testParamLarge(t);
}

// CIR-LABEL: cir.func {{.*}} @_Z14testCallLarge0v
// CIR:   cir.call @_ZN5LargeC1Ev
// CIR:   cir.call @_ZN5LargeC1ERKS_
// CIR:   cir.call @_Z14testParamLarge5Large
// LLVM-LABEL: define {{.*}} void @_Z14testCallLarge0v(
// LLVM:   call void @_ZN5LargeC1Ev(
// LLVM:   call void @_ZN5LargeC1ERKS_(
// LLVM:   call void @_Z14testParamLarge5Large(
// OGCG-LABEL: define {{.*}} void @_Z14testCallLarge0v(
// OGCG:   call {{.*}} @_ZN5LargeC1Ev(
// OGCG:   call {{.*}} @_ZN5LargeC1ERKS_(
// OGCG:   call void @_Z14testParamLarge5Large(

// --- Case 10: testCallLarge1 (pass returned value directly) ---

void testCallLarge1() {
  testParamLarge(testReturnLarge());
}

// CIR-LABEL: cir.func {{.*}} @_Z14testCallLarge1v
// CIR:   cir.call @_Z15testReturnLargev
// CIR:   cir.call @_Z14testParamLarge5Large
// LLVM-LABEL: define {{.*}} void @_Z14testCallLarge1v(
// LLVM:   call {{.*}} @_Z15testReturnLargev(
// LLVM:   call void @_Z14testParamLarge5Large(
// OGCG-LABEL: define {{.*}} void @_Z14testCallLarge1v(
// OGCG:   call void @_Z15testReturnLargev(
// OGCG:   call void @_Z14testParamLarge5Large(

// --- Case 11: testIgnoredLarge (discard return, must destruct) ---

void testIgnoredLarge() {
  testReturnLarge();
}

// CIR-LABEL: cir.func {{.*}} @_Z16testIgnoredLargev
// CIR:   cir.call @_Z15testReturnLargev
// CIR:   cir.call @_ZN5LargeD1Ev
// LLVM-LABEL: define {{.*}} void @_Z16testIgnoredLargev(
// LLVM:   call {{.*}} @_Z15testReturnLargev(
// LLVM:   call void @_ZN5LargeD1Ev(
// OGCG-LABEL: define {{.*}} void @_Z16testIgnoredLargev(
// OGCG:   call void @_Z15testReturnLargev(
// OGCG:   call {{.*}} @_ZN5LargeD1Ev(

// --- Case 12: testReturnHasTrivial ---

Trivial testReturnHasTrivial() {
  Trivial t;
  return t;
}

// CIR-LABEL: cir.func {{.*}} @_Z20testReturnHasTrivialv
// CIR:   cir.return
// LLVM-LABEL: define {{.*}} @_Z20testReturnHasTrivialv(
// LLVM:   ret
// OGCG-LABEL: define {{.*}} i32 @_Z20testReturnHasTrivialv(
// OGCG:   ret i32

// --- Case 13: testReturnHasNonTrivial ---

NonTrivial testReturnHasNonTrivial() {
  NonTrivial t;
  return t;
}

// CIR-LABEL: cir.func {{.*}} @_Z23testReturnHasNonTrivialv
// CIR:   cir.call @_ZN10NonTrivialC1Ev
// LLVM-LABEL: define {{.*}} @_Z23testReturnHasNonTrivialv(
// LLVM:   call void @_ZN10NonTrivialC1Ev(
// OGCG-LABEL: define {{.*}} void @_Z23testReturnHasNonTrivialv(ptr {{.*}}sret(%struct.NonTrivial)
// OGCG:   call {{.*}} @_ZN10NonTrivialC1Ev(

// --- Case 14: testExceptionSmall ---
// CIR does not emit invoke/landingpad; the non-unwinding call sequence
// is tested here.  OGCG verifies the full EH path.

void calleeExceptionSmall(Small, Small);

void testExceptionSmall() {
  calleeExceptionSmall(Small(), Small());
}

// CIR-LABEL: cir.func {{.*}} @_Z18testExceptionSmallv
// CIR:   cir.call @_ZN5SmallC1Ev
// CIR:   cir.call @_ZN5SmallC1Ev
// CIR:   cir.call @_Z20calleeExceptionSmall5SmallS_
// LLVM-LABEL: define {{.*}} void @_Z18testExceptionSmallv(
// LLVM:   call void @_ZN5SmallC1Ev(
// LLVM:   call void @_ZN5SmallC1Ev(
// LLVM:   call void @_Z20calleeExceptionSmall5SmallS_(
// OGCG-LABEL: define {{.*}} void @_Z18testExceptionSmallv()
// OGCG:   call {{.*}} @_ZN5SmallC1Ev(
// OGCG:   call {{.*}} @_ZN5SmallC1Ev(
// OGCG:   call void @_Z20calleeExceptionSmall5SmallS_(

// --- Case 15: testExceptionLarge ---

void calleeExceptionLarge(Large, Large);

void testExceptionLarge() {
  calleeExceptionLarge(Large(), Large());
}

// CIR-LABEL: cir.func {{.*}} @_Z18testExceptionLargev
// CIR:   cir.call @_ZN5LargeC1Ev
// CIR:   cir.call @_ZN5LargeC1Ev
// CIR:   cir.call @_Z20calleeExceptionLarge5LargeS_
// LLVM-LABEL: define {{.*}} void @_Z18testExceptionLargev(
// LLVM:   call void @_ZN5LargeC1Ev(
// LLVM:   call void @_ZN5LargeC1Ev(
// LLVM:   call void @_Z20calleeExceptionLarge5LargeS_(
// OGCG-LABEL: define {{.*}} void @_Z18testExceptionLargev()
// OGCG:   call {{.*}} @_ZN5LargeC1Ev(
// OGCG:   call {{.*}} @_ZN5LargeC1Ev(
// OGCG:   call void @_Z20calleeExceptionLarge5LargeS_(

// --- Case 16: GH93040 packed trivial_abi with placement new ---

void* operator new(unsigned long, void*);
namespace GH93040 {
struct [[clang::trivial_abi]] S {
  char a;
  int x;
  __attribute((aligned(2))) char y;
  S();
} __attribute((packed));
S f();
void g(S* s) { new(s) S(f()); }
}

// CIR-LABEL: cir.func {{.*}} @_ZN7GH930401gEPNS_1SE
// CIR:   cir.call @_ZN7GH930401fEv
// LLVM-LABEL: define {{.*}} void @_ZN7GH930401gEPNS_1SE(
// LLVM:   call {{.*}} @_ZN7GH930401fEv(
// OGCG-LABEL: define {{.*}} void @_ZN7GH930401gEPNS_1SE(
// OGCG:   call void @_ZN7GH930401fEv(ptr {{.*}}sret(

// Lambda __invoke comes last in OGCG (internal linkage).
// OGCG-LABEL: define {{.*}} @{{.*}}__invokeEv
// OGCG:   call {{.*}} @{{.*}}clEv
