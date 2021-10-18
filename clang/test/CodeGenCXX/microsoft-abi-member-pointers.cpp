// RUN: %clang_cc1 -std=c++11 -Wno-uninitialized -fno-rtti -emit-llvm %s -o - -triple=i386-pc-win32 -fms-extensions | FileCheck -allow-deprecated-dag-overlap %s
// RUN: %clang_cc1 -std=c++11 -Wno-uninitialized -fno-rtti -emit-llvm %s -o - -triple=x86_64-pc-win32 -fms-extensions | FileCheck %s -check-prefix=X64
// RUN: %clang_cc1 -std=c++11 -Wno-uninitialized -fno-rtti -emit-llvm %s -o - -triple=i386-pc-win32 -DINCOMPLETE_VIRTUAL -fms-extensions -verify
// RUN: %clang_cc1 -std=c++11 -Wno-uninitialized -fno-rtti -emit-llvm %s -o - -triple=i386-pc-win32 -DINCOMPLETE_VIRTUAL -DMEMFUN -fms-extensions -verify

namespace pr37399 {
template <typename T>
struct Functor {
  void (T::*PtrToMemberFunction)();
};
// CHECK-DAG: %"struct.pr37399::Functor" = type { i8* }

template <typename SomeType>
class SimpleDerivedFunctor;
template <typename SomeType>
class SimpleDerivedFunctor : public Functor<SimpleDerivedFunctor<SomeType>> {};
// CHECK-DAG: %"class.pr37399::SimpleDerivedFunctor" = type { %"struct.pr37399::Functor" }

SimpleDerivedFunctor<void> SimpleFunctor;
// CHECK-DAG: @"?SimpleFunctor@pr37399@@3V?$SimpleDerivedFunctor@X@1@A" = dso_local global %"class.pr37399::SimpleDerivedFunctor" zeroinitializer, align 4

short Global = 0;
template <typename SomeType>
class DerivedFunctor;
template <typename SomeType>
class DerivedFunctor
    : public Functor<DerivedFunctor<void>> {
public:
  void Foo() {
    Global = 42;
  }
};

class MultipleBase {
public:
  MultipleBase() : Value() {}
  short Value;
};
// CHECK-DAG: %"class.pr37399::MultipleBase" = type { i16 }

template <typename SomeType>
class MultiplyDerivedFunctor;
template <typename SomeType>
class MultiplyDerivedFunctor
    : public Functor<MultiplyDerivedFunctor<void>>,
      public MultipleBase {
public:
  void Foo() {
    MultipleBase::Value = 42*2;
  }
};

class VirtualBase {
public:
  VirtualBase() : Value() {}
  short Value;
};
// CHECK-DAG: %"class.pr37399::VirtualBase" = type { i16 }

template <typename SomeType>
class VirtBaseFunctor
    : public Functor<SomeType>,
      public virtual VirtualBase{};
template <typename SomeType>
class VirtuallyDerivedFunctor;
template <typename SomeType>
class VirtuallyDerivedFunctor
    : public VirtBaseFunctor<VirtuallyDerivedFunctor<void>>,
      public virtual VirtualBase {
public:
  void Foo() {
    VirtualBase::Value = 42*3;
  }
};
} // namespace pr37399

pr37399::DerivedFunctor<int>           BFunctor;
// CHECK-DAG: @"?BFunctor@@3V?$DerivedFunctor@H@pr37399@@A" = dso_local global %"[[BFUNCTOR:class.pr37399::DerivedFunctor(\.[0-9]+)?]]" zeroinitializer, align 8
// CHECK-DAG: %"[[BFUNCTOR]]" = type { %"[[BFUNCTORBASE:struct.pr37399::Functor(\.[0-9]+)?]]" }
// CHECK-DAG: %"[[BFUNCTORBASE]]" = type { { i8*, i32, i32, i32 } }
pr37399::DerivedFunctor<void>          AFunctor;
// CHECK-DAG: @"?AFunctor@@3V?$DerivedFunctor@X@pr37399@@A" = dso_local global %"[[AFUNCTOR:class.pr37399::DerivedFunctor(\.[0-9]+)?]]" zeroinitializer, align 8
// CHECK-DAG: %"[[AFUNCTOR]]" = type { %"[[AFUNCTORBASE:struct.pr37399::Functor(\.[0-9]+)?]]" }
// CHECK-DAG: %"[[AFUNCTORBASE]]" = type { { i8*, i32, i32, i32 } }

pr37399::MultiplyDerivedFunctor<int>   DFunctor;
// CHECK-DAG: @"?DFunctor@@3V?$MultiplyDerivedFunctor@H@pr37399@@A" = dso_local global %"[[DFUNCTOR:class.pr37399::MultiplyDerivedFunctor(\.[0-9]+)?]]" zeroinitializer, align 8
// CHECK-DAG: %"[[DFUNCTOR]]" = type { %"[[DFUNCTORBASE:struct.pr37399::Functor(\.[0-9]+)?]]", %"class.pr37399::MultipleBase", [6 x i8] }
// CHECK-DAG: %"[[DFUNCTORBASE]]" = type { { i8*, i32, i32, i32 } }
pr37399::MultiplyDerivedFunctor<void>  CFunctor;
// CHECK-DAG: @"?CFunctor@@3V?$MultiplyDerivedFunctor@X@pr37399@@A" = dso_local global %"[[CFUNCTOR:class.pr37399::MultiplyDerivedFunctor(\.[0-9]+)?]]" zeroinitializer, align 8
// CHECK-DAG: %"[[CFUNCTOR]]" = type { %"[[CFUNCTORBASE:struct.pr37399::Functor(\.[0-9]+)?]]", %"class.pr37399::MultipleBase", [6 x i8] }
// CHECK-DAG: %"[[CFUNCTORBASE]]" = type { { i8*, i32, i32, i32 } }

pr37399::VirtuallyDerivedFunctor<int>  FFunctor;
// CHECK-DAG: @"?FFunctor@@3V?$VirtuallyDerivedFunctor@H@pr37399@@A" = dso_local global %"[[FFUNCTOR:class.pr37399::VirtuallyDerivedFunctor(\.[0-9]+)?]]" zeroinitializer, align 8
// CHECK-DAG: %"[[FFUNCTOR]]" = type { %"class.pr37399::VirtBaseFunctor.base", %"class.pr37399::VirtualBase" }
pr37399::VirtuallyDerivedFunctor<void> EFunctor;
// CHECK-DAG: @"?EFunctor@@3V?$VirtuallyDerivedFunctor@X@pr37399@@A" = dso_local global %"[[EFUNCTOR:class.pr37399::VirtuallyDerivedFunctor(\.[0-9]+)?]]" zeroinitializer, align 8
// CHECK-DAG: %"[[EFUNCTOR]]" = type { %"class.pr37399::VirtBaseFunctor.base", %"class.pr37399::VirtualBase" }

// CHECK-DAG: %"class.pr37399::VirtBaseFunctor.base" = type <{ %"[[VFUNCTORBASE:struct.pr37399::Functor(\.[0-9]+)?]]", i32*, [4 x i8] }>
// CHECK-DAG: %"[[VFUNCTORBASE]]" = type { { i8*, i32, i32, i32 } }

namespace pr37399 {
void SingleInheritanceFnPtrCall() {
  BFunctor.PtrToMemberFunction = &DerivedFunctor<void>::Foo;
  (AFunctor.*(BFunctor.PtrToMemberFunction))();
}
void MultipleInheritanceFnPtrCall() {
  DFunctor.PtrToMemberFunction = &MultiplyDerivedFunctor<void>::Foo;
  Global = CFunctor.MultipleBase::Value;
  (CFunctor.*(DFunctor.PtrToMemberFunction))();
  Global = CFunctor.MultipleBase::Value;
}
void VirtualInheritanceFnPtrCall() {
  FFunctor.PtrToMemberFunction = &VirtuallyDerivedFunctor<void>::Foo;
  Global = EFunctor.VirtualBase::Value;
  (EFunctor.*(FFunctor.PtrToMemberFunction))();
  Global = EFunctor.VirtualBase::Value;
}
} // namespace pr37399

namespace pr43803 {
// This case is interesting because it exercises conversion between member
// pointer types when emitting constants.

struct B;
struct C { int B::*option; };
extern const C table[3];
struct A {
  int x, y;
  // Test the indirect case.
  struct {
    int z;
  };
};
struct B : A {};
const C table[] = {
    {&B::x},
    {&B::y},
    {&B::z},
};

// CHECK: @"?table@pr43803@@3QBUC@1@B" = dso_local constant [3 x %"struct.pr43803::C"]
// CHECK-SAME: [%"struct.pr43803::C" { { i32, i32, i32 } zeroinitializer, [4 x i8] undef },
// CHECK-SAME:  %"struct.pr43803::C" { { i32, i32, i32 } { i32 4, i32 0, i32 0 }, [4 x i8] undef },
// CHECK-SAME:  %"struct.pr43803::C" { { i32, i32, i32 } { i32 8, i32 0, i32 0 }, [4 x i8] undef }]
}

namespace pr48687 {
template <typename T> struct A {
  T value;
  static constexpr auto address = &A<T>::value;
};
extern template class A<float>;
template class A<float>;
// CHECK: @"?address@?$A@M@pr48687@@2QQ12@MQ12@" = weak_odr dso_local constant i32 0, comdat, align 4
}

struct PR26313_Y;
typedef void (PR26313_Y::*PR26313_FUNC)();
struct PR26313_X {
  PR26313_FUNC *ptr;
  PR26313_X();
};
PR26313_X::PR26313_X() {}
void PR26313_f(PR26313_FUNC *p) { delete p; }

struct PR26313_Z;
int PR26313_Z::**a = nullptr;
int PR26313_Z::*b = *a;
// CHECK-DAG: @"?a@@3PAPQPR26313_Z@@HA" = dso_local global %0* null, align 4
// CHECK-DAG: @"?b@@3PQPR26313_Z@@HQ1@" = dso_local global { i32, i32, i32 } { i32 0, i32 0, i32 -1 }, align 4

namespace PR20947 {
struct A;
int A::**a = nullptr;
// CHECK-DAG: @"?a@PR20947@@3PAPQA@1@HA" = dso_local global %{{.*}}* null, align 4

struct B;
int B::*&b = b;
// CHECK-DAG: @"?b@PR20947@@3AAPQB@1@HA" = dso_local global %{{.*}}* null, align 4
}

namespace PR20017 {
template <typename T>
struct A {
  int T::*m_fn1() { return nullptr; }
};
struct B;
auto a = &A<B>::m_fn1;
// CHECK-DAG: @"?a@PR20017@@3P8?$A@UB@PR20017@@@1@AEPQB@1@HXZQ21@" = dso_local global i8* bitcast ({ i32, i32, i32 } ({{.*}}*)* @"?m_fn1@?$A@UB@PR20017@@@PR20017@@QAEPQB@2@HXZ" to i8*), align 4
}

#ifndef INCOMPLETE_VIRTUAL
struct B1 {
  void foo();
  int b;
};
struct B2 {
  int b2;
  void foo();
};
struct Single : B1 {
  void foo();
};
struct Multiple : B1, B2 {
  int m;
  void foo();
};
struct Virtual : virtual B1 {
  int v;
  void foo();
};

struct POD {
  int a;
  int b;
};

struct Polymorphic {
  virtual void myVirtual();
  int a;
  int b;
};

// This class uses the virtual inheritance model, yet its vbptr offset is not 0.
// We still use zero for the null field offset, despite it being a valid field
// offset.
struct NonZeroVBPtr : POD, Virtual {
  int n;
  void foo();
};

struct Unspecified;
struct UnspecSingle;

// Check that we can lower the LLVM types and get the null initializers right.
int Single     ::*s_d_memptr;
int Polymorphic::*p_d_memptr;
int Multiple   ::*m_d_memptr;
int Virtual    ::*v_d_memptr;
int NonZeroVBPtr::*n_d_memptr;
int Unspecified::*u_d_memptr;
int UnspecSingle::*us_d_memptr;
// CHECK: @"?s_d_memptr@@3PQSingle@@HQ1@" = dso_local global i32 -1, align 4
// CHECK: @"?p_d_memptr@@3PQPolymorphic@@HQ1@" = dso_local global i32 0, align 4
// CHECK: @"?m_d_memptr@@3PQMultiple@@HQ1@" = dso_local global i32 -1, align 4
// CHECK: @"?v_d_memptr@@3PQVirtual@@HQ1@" = dso_local global { i32, i32 }
// CHECK:   { i32 0, i32 -1 }, align 4
// CHECK: @"?n_d_memptr@@3PQNonZeroVBPtr@@HQ1@" = dso_local global { i32, i32 }
// CHECK:   { i32 0, i32 -1 }, align 4
// CHECK: @"?u_d_memptr@@3PQUnspecified@@HQ1@" = dso_local global { i32, i32, i32 }
// CHECK:   { i32 0, i32 0, i32 -1 }, align 4
// CHECK: @"?us_d_memptr@@3PQUnspecSingle@@HQ1@" = dso_local global { i32, i32, i32 }
// CHECK:   { i32 0, i32 0, i32 -1 }, align 4

void (Single  ::*s_f_memptr)();
void (Multiple::*m_f_memptr)();
void (Virtual ::*v_f_memptr)();
// CHECK: @"?s_f_memptr@@3P8Single@@AEXXZQ1@" = dso_local global i8* null, align 4
// CHECK: @"?m_f_memptr@@3P8Multiple@@AEXXZQ1@" = dso_local global { i8*, i32 } zeroinitializer, align 4
// CHECK: @"?v_f_memptr@@3P8Virtual@@AEXXZQ1@" = dso_local global { i8*, i32, i32 } zeroinitializer, align 4

// We can define Unspecified after locking in the inheritance model.
struct Unspecified : Multiple, Virtual {
  void foo();
  int u;
};

struct UnspecSingle {
  void foo();
};

// Test memptr emission in a constant expression.
namespace Const {
void (Single     ::*s_f_mp)() = &Single::foo;
void (Multiple   ::*m_f_mp)() = &B2::foo;
void (Virtual    ::*v_f_mp)() = &Virtual::foo;
void (Unspecified::*u_f_mp)() = &Unspecified::foo;
void (UnspecSingle::*us_f_mp)() = &UnspecSingle::foo;
// CHECK: @"?s_f_mp@Const@@3P8Single@@AEXXZQ2@" =
// CHECK:   global i8* bitcast ({{.*}} @"?foo@Single@@QAEXXZ" to i8*), align 4
// CHECK: @"?m_f_mp@Const@@3P8Multiple@@AEXXZQ2@" =
// CHECK:   global { i8*, i32 } { i8* bitcast ({{.*}} @"?foo@B2@@QAEXXZ" to i8*), i32 4 }, align 4
// CHECK: @"?v_f_mp@Const@@3P8Virtual@@AEXXZQ2@" =
// CHECK:   global { i8*, i32, i32 } { i8* bitcast ({{.*}} @"?foo@Virtual@@QAEXXZ" to i8*), i32 0, i32 0 }, align 4
// CHECK: @"?u_f_mp@Const@@3P8Unspecified@@AEXXZQ2@" =
// CHECK:   global { i8*, i32, i32, i32 } { i8* bitcast ({{.*}} @"?foo@Unspecified@@QAEXXZ" to i8*), i32 0, i32 0, i32 0 }, align 4
// CHECK: @"?us_f_mp@Const@@3P8UnspecSingle@@AEXXZQ2@" =
// CHECK:   global { i8*, i32, i32, i32 } { i8* bitcast ({{.*}} @"?foo@UnspecSingle@@QAEXXZ" to i8*), i32 0, i32 0, i32 0 }, align 4
}

namespace CastParam {
// This exercises ConstExprEmitter instead of ValueDecl::evaluateValue.  The
// extra reinterpret_cast for the parameter type requires more careful folding.
// FIXME: Or does it?  If reinterpret_casts are no-ops, we should be able to
// strip them in evaluateValue() and just proceed as normal with an APValue.
struct A {
  int a;
  void foo(A *p);
};
struct B { int b; };
struct C : B, A { int c; };

void (A::*ptr1)(void *) = (void (A::*)(void *)) &A::foo;
// CHECK: @"?ptr1@CastParam@@3P8A@1@AEXPAX@ZQ21@" =
// CHECK:   global i8* bitcast (void ({{.*}})* @"?foo@A@CastParam@@QAEXPAU12@@Z" to i8*), align 4

// Try a reinterpret_cast followed by a memptr conversion.
void (C::*ptr2)(void *) = (void (C::*)(void *)) (void (A::*)(void *)) &A::foo;
// CHECK: @"?ptr2@CastParam@@3P8C@1@AEXPAX@ZQ21@" =
// CHECK:   global { i8*, i32 } { i8* bitcast (void ({{.*}})* @"?foo@A@CastParam@@QAEXPAU12@@Z" to i8*), i32 4 }, align 4

void (C::*ptr3)(void *) = (void (C::*)(void *)) (void (A::*)(void *)) (void (A::*)(A *)) 0;
// CHECK: @"?ptr3@CastParam@@3P8C@1@AEXPAX@ZQ21@" =
// CHECK:   global { i8*, i32 } zeroinitializer, align 4

struct D : C {
  virtual void isPolymorphic();
  int d;
};

// Try a cast that changes the inheritance model.  Null for D is 0, but null for
// C is -1.  We need the cast to long in order to hit the non-APValue path.
int C::*ptr4 = (int C::*) (int D::*) (long D::*) 0;
// CHECK: @"?ptr4@CastParam@@3PQC@1@HQ21@" = dso_local global i32 -1, align 4

// MSVC rejects this but we accept it.
int C::*ptr5 = (int C::*) (long D::*) 0;
// CHECK: @"?ptr5@CastParam@@3PQC@1@HQ21@" = dso_local global i32 -1, align 4
}

struct UnspecWithVBPtr;
int UnspecWithVBPtr::*forceUnspecWithVBPtr;
struct UnspecWithVBPtr : B1, virtual B2 {
  int u;
  void foo();
};

// Test emitting non-virtual member pointers in a non-constexpr setting.
void EmitNonVirtualMemberPointers() {
  void (Single     ::*s_f_memptr)() = &Single::foo;
  void (Multiple   ::*m_f_memptr)() = &Multiple::foo;
  void (Virtual    ::*v_f_memptr)() = &Virtual::foo;
  void (Unspecified::*u_f_memptr)() = &Unspecified::foo;
  void (UnspecWithVBPtr::*u2_f_memptr)() = &UnspecWithVBPtr::foo;
// CHECK: define dso_local void @"?EmitNonVirtualMemberPointers@@YAXXZ"() {{.*}} {
// CHECK:   alloca i8*, align 4
// CHECK:   alloca { i8*, i32 }, align 4
// CHECK:   alloca { i8*, i32, i32 }, align 4
// CHECK:   alloca { i8*, i32, i32, i32 }, align 4
// CHECK:   store i8* bitcast (void (%{{.*}}*)* @"?foo@Single@@QAEXXZ" to i8*), i8** %{{.*}}, align 4
// CHECK:   store { i8*, i32 }
// CHECK:     { i8* bitcast (void (%{{.*}}*)* @"?foo@Multiple@@QAEXXZ" to i8*), i32 0 },
// CHECK:     { i8*, i32 }* %{{.*}}, align 4
// CHECK:   store { i8*, i32, i32 }
// CHECK:     { i8* bitcast (void (%{{.*}}*)* @"?foo@Virtual@@QAEXXZ" to i8*), i32 0, i32 0 },
// CHECK:     { i8*, i32, i32 }* %{{.*}}, align 4
// CHECK:   store { i8*, i32, i32, i32 }
// CHECK:     { i8* bitcast (void (%{{.*}}*)* @"?foo@Unspecified@@QAEXXZ" to i8*), i32 0, i32 0, i32 0 },
// CHECK:     { i8*, i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   store { i8*, i32, i32, i32 }
// CHECK:     { i8* bitcast (void (%{{.*}}*)* @"?foo@UnspecWithVBPtr@@QAEXXZ" to i8*),
// CHECK:       i32 0, i32 0, i32 0 },
// CHECK:     { i8*, i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   ret void
// CHECK: }
}

void podMemPtrs() {
  int POD::*memptr;
  memptr = &POD::a;
  memptr = &POD::b;
  if (memptr)
    memptr = 0;
// Check that member pointers use the right offsets and that null is -1.
// CHECK:      define dso_local void @"?podMemPtrs@@YAXXZ"() {{.*}} {
// CHECK:        %[[memptr:.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32 0, i32* %[[memptr]], align 4
// CHECK-NEXT:   store i32 4, i32* %[[memptr]], align 4
// CHECK-NEXT:   %[[memptr_val:.*]] = load i32, i32* %[[memptr]], align 4
// CHECK-NEXT:   %{{.*}} = icmp ne i32 %[[memptr_val]], -1
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK:        store i32 -1, i32* %[[memptr]], align 4
// CHECK:        ret void
// CHECK:      }
}

void polymorphicMemPtrs() {
  int Polymorphic::*memptr;
  memptr = &Polymorphic::a;
  memptr = &Polymorphic::b;
  if (memptr)
    memptr = 0;
// Member pointers for polymorphic classes include the vtable slot in their
// offset and use 0 to represent null.
// CHECK:      define dso_local void @"?polymorphicMemPtrs@@YAXXZ"() {{.*}} {
// CHECK:        %[[memptr:.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32 4, i32* %[[memptr]], align 4
// CHECK-NEXT:   store i32 8, i32* %[[memptr]], align 4
// CHECK-NEXT:   %[[memptr_val:.*]] = load i32, i32* %[[memptr]], align 4
// CHECK-NEXT:   %{{.*}} = icmp ne i32 %[[memptr_val]], 0
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK:        store i32 0, i32* %[[memptr]], align 4
// CHECK:        ret void
// CHECK:      }
}

bool nullTestDataUnspecified(int Unspecified::*mp) {
  return mp;
// CHECK: define dso_local zeroext i1 @"?nullTestDataUnspecified@@YA_NPQUnspecified@@H@Z"{{.*}} {
// CHECK:   %{{.*}} = load { i32, i32, i32 }, { i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   store { i32, i32, i32 } {{.*}} align 4
// CHECK:   %[[mp:.*]] = load { i32, i32, i32 }, { i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   %[[mp0:.*]] = extractvalue { i32, i32, i32 } %[[mp]], 0
// CHECK:   %[[cmp0:.*]] = icmp ne i32 %[[mp0]], 0
// CHECK:   %[[mp1:.*]] = extractvalue { i32, i32, i32 } %[[mp]], 1
// CHECK:   %[[cmp1:.*]] = icmp ne i32 %[[mp1]], 0
// CHECK:   %[[and0:.*]] = or i1 %[[cmp0]], %[[cmp1]]
// CHECK:   %[[mp2:.*]] = extractvalue { i32, i32, i32 } %[[mp]], 2
// CHECK:   %[[cmp2:.*]] = icmp ne i32 %[[mp2]], -1
// CHECK:   %[[and1:.*]] = or i1 %[[and0]], %[[cmp2]]
// CHECK:   ret i1 %[[and1]]
// CHECK: }

// Pass this large type indirectly.
// X64-LABEL: define dso_local zeroext i1 @"?nullTestDataUnspecified@@
// X64:             ({ i32, i32, i32 }* %0)
}

bool nullTestFunctionUnspecified(void (Unspecified::*mp)()) {
  return mp;
// CHECK: define dso_local zeroext i1 @"?nullTestFunctionUnspecified@@YA_NP8Unspecified@@AEXXZ@Z"{{.*}} {
// CHECK:   %{{.*}} = load { i8*, i32, i32, i32 }, { i8*, i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   store { i8*, i32, i32, i32 } {{.*}} align 4
// CHECK:   %[[mp:.*]] = load { i8*, i32, i32, i32 }, { i8*, i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   %[[mp0:.*]] = extractvalue { i8*, i32, i32, i32 } %[[mp]], 0
// CHECK:   %[[cmp0:.*]] = icmp ne i8* %[[mp0]], null
// CHECK:   ret i1 %[[cmp0]]
// CHECK: }
}

int loadDataMemberPointerVirtual(Virtual *o, int Virtual::*memptr) {
  return o->*memptr;
// Test that we can unpack this aggregate member pointer and load the member
// data pointer.
// CHECK: define dso_local i32 @"?loadDataMemberPointerVirtual@@YAHPAUVirtual@@PQ1@H@Z"{{.*}} {
// CHECK:   %[[o:.*]] = load %{{.*}}*, %{{.*}}** %{{.*}}, align 4
// CHECK:   %[[memptr:.*]] = load { i32, i32 }, { i32, i32 }* %{{.*}}, align 4
// CHECK:   %[[memptr0:.*]] = extractvalue { i32, i32 } %[[memptr:.*]], 0
// CHECK:   %[[memptr1:.*]] = extractvalue { i32, i32 } %[[memptr:.*]], 1
// CHECK:   %[[v6:.*]] = bitcast %{{.*}}* %[[o]] to i8*
// CHECK:   %[[vbptr:.*]] = getelementptr inbounds i8, i8* %[[v6]], i32 0
// CHECK:   %[[vbptr_a:.*]] = bitcast i8* %[[vbptr]] to i32**
// CHECK:   %[[vbtable:.*]] = load i32*, i32** %[[vbptr_a:.*]]
// CHECK:   %[[memptr1_shr:.*]] = ashr exact i32 %[[memptr1]], 2
// CHECK:   %[[v7:.*]] = getelementptr inbounds i32, i32* %[[vbtable]], i32 %[[memptr1_shr]]
// CHECK:   %[[vbase_offs:.*]] = load i32, i32* %[[v7]]
// CHECK:   %[[v10:.*]] = getelementptr inbounds i8, i8* %[[vbptr]], i32 %[[vbase_offs]]
// CHECK:   %[[offset:.*]] = getelementptr inbounds i8, i8* %[[v10]], i32 %[[memptr0]]
// CHECK:   %[[v11:.*]] = bitcast i8* %[[offset]] to i32*
// CHECK:   %[[v12:.*]] = load i32, i32* %[[v11]]
// CHECK:   ret i32 %[[v12]]
// CHECK: }

// A two-field data memptr on x64 gets coerced to i64 and is passed in a
// register or memory.
// X64-LABEL: define dso_local i32 @"?loadDataMemberPointerVirtual@@YAHPEAUVirtual@@PEQ1@H@Z"
// X64:             (%struct.Virtual* %o, i64 %memptr.coerce)
}

int loadDataMemberPointerUnspecified(Unspecified *o, int Unspecified::*memptr) {
  return o->*memptr;
// Test that we can unpack this aggregate member pointer and load the member
// data pointer.
// CHECK: define dso_local i32 @"?loadDataMemberPointerUnspecified@@YAHPAUUnspecified@@PQ1@H@Z"{{.*}} {
// CHECK:   %[[o:.*]] = load %{{.*}}*, %{{.*}}** %{{.*}}, align 4
// CHECK:   %[[memptr:.*]] = load { i32, i32, i32 }, { i32, i32, i32 }* %{{.*}}, align 4
// CHECK:   %[[memptr0:.*]] = extractvalue { i32, i32, i32 } %[[memptr:.*]], 0
// CHECK:   %[[memptr1:.*]] = extractvalue { i32, i32, i32 } %[[memptr:.*]], 1
// CHECK:   %[[memptr2:.*]] = extractvalue { i32, i32, i32 } %[[memptr:.*]], 2
// CHECK:   %[[base:.*]] = bitcast %{{.*}}* %[[o]] to i8*
// CHECK:   %[[is_vbase:.*]] = icmp ne i32 %[[memptr2]], 0
// CHECK:   br i1 %[[is_vbase]], label %[[vadjust:.*]], label %[[skip:.*]]
//
// CHECK: [[vadjust]]
// CHECK:   %[[vbptr:.*]] = getelementptr inbounds i8, i8* %[[base]], i32 %[[memptr1]]
// CHECK:   %[[vbptr_a:.*]] = bitcast i8* %[[vbptr]] to i32**
// CHECK:   %[[vbtable:.*]] = load i32*, i32** %[[vbptr_a:.*]]
// CHECK:   %[[memptr2_shr:.*]] = ashr exact i32 %[[memptr2]], 2
// CHECK:   %[[v7:.*]] = getelementptr inbounds i32, i32* %[[vbtable]], i32 %[[memptr2_shr]]
// CHECK:   %[[vbase_offs:.*]] = load i32, i32* %[[v7]]
// CHECK:   %[[base_adj:.*]] = getelementptr inbounds i8, i8* %[[vbptr]], i32 %[[vbase_offs]]
//
// CHECK: [[skip]]
// CHECK:   %[[new_base:.*]] = phi i8* [ %[[base]], %{{.*}} ], [ %[[base_adj]], %[[vadjust]] ]
// CHECK:   %[[offset:.*]] = getelementptr inbounds i8, i8* %[[new_base]], i32 %[[memptr0]]
// CHECK:   %[[v11:.*]] = bitcast i8* %[[offset]] to i32*
// CHECK:   %[[v12:.*]] = load i32, i32* %[[v11]]
// CHECK:   ret i32 %[[v12]]
// CHECK: }
}

void callMemberPointerSingle(Single *o, void (Single::*memptr)()) {
  (o->*memptr)();
// Just look for an indirect thiscall.
// CHECK: define dso_local void @"?callMemberPointerSingle@@{{.*}} {{.*}} {
// CHECK:   call x86_thiscallcc void %{{.*}}(%{{.*}} %{{.*}})
// CHECK:   ret void
// CHECK: }

// X64-LABEL: define dso_local void @"?callMemberPointerSingle@@
// X64:           (%struct.Single* %o, i8* %memptr)
// X64:   bitcast i8* %{{[^ ]*}} to void (%struct.Single*)*
// X64:   ret void
}

void callMemberPointerMultiple(Multiple *o, void (Multiple::*memptr)()) {
  (o->*memptr)();
// CHECK: define dso_local void @"?callMemberPointerMultiple@@{{.*}} {
// CHECK:   %[[memptr0:.*]] = extractvalue { i8*, i32 } %{{.*}}, 0
// CHECK:   %[[memptr1:.*]] = extractvalue { i8*, i32 } %{{.*}}, 1
// CHECK:   %[[this_adjusted:.*]] = getelementptr inbounds i8, i8* %{{.*}}, i32 %[[memptr1]]
// CHECK:   %[[this:.*]] = bitcast i8* %[[this_adjusted]] to {{.*}}
// CHECK:   %[[fptr:.*]] = bitcast i8* %[[memptr0]] to {{.*}}
// CHECK:   call x86_thiscallcc void %[[fptr]](%{{.*}} %[[this]])
// CHECK:   ret void
// CHECK: }
}

void callMemberPointerVirtualBase(Virtual *o, void (Virtual::*memptr)()) {
  (o->*memptr)();
// This shares a lot with virtual data member pointers.
// CHECK: define dso_local void @"?callMemberPointerVirtualBase@@{{.*}} {
// CHECK:   %[[memptr0:.*]] = extractvalue { i8*, i32, i32 } %{{.*}}, 0
// CHECK:   %[[memptr1:.*]] = extractvalue { i8*, i32, i32 } %{{.*}}, 1
// CHECK:   %[[memptr2:.*]] = extractvalue { i8*, i32, i32 } %{{.*}}, 2
// CHECK:   %[[vbptr:.*]] = getelementptr inbounds i8, i8* %{{.*}}, i32 0
// CHECK:   %[[vbptr_a:.*]] = bitcast i8* %[[vbptr]] to i32**
// CHECK:   %[[vbtable:.*]] = load i32*, i32** %[[vbptr_a:.*]]
// CHECK:   %[[memptr2_shr:.*]] = ashr exact i32 %[[memptr2]], 2
// CHECK:   %[[v7:.*]] = getelementptr inbounds i32, i32* %[[vbtable]], i32 %[[memptr2_shr]]
// CHECK:   %[[vbase_offs:.*]] = load i32, i32* %[[v7]]
// CHECK:   %[[v10:.*]] = getelementptr inbounds i8, i8* %[[vbptr]], i32 %[[vbase_offs]]
// CHECK:   %[[this_adjusted:.*]] = getelementptr inbounds i8, i8* %[[v10]], i32 %[[memptr1]]
// CHECK:   %[[fptr:.*]] = bitcast i8* %[[memptr0]] to void ({{.*}})
// CHECK:   %[[this:.*]] = bitcast i8* %[[this_adjusted]] to {{.*}}
// CHECK:   call x86_thiscallcc void %[[fptr]](%{{.*}} %[[this]])
// CHECK:   ret void
// CHECK: }
}

bool compareSingleFunctionMemptr(void (Single::*l)(), void (Single::*r)()) {
  return l == r;
// Should only be one comparison here.
// CHECK: define dso_local zeroext i1 @"?compareSingleFunctionMemptr@@YA_NP8Single@@AEXXZ0@Z"{{.*}} {
// CHECK-NOT: icmp
// CHECK:   %[[r:.*]] = icmp eq
// CHECK-NOT: icmp
// CHECK:   ret i1 %[[r]]
// CHECK: }

// X64-LABEL: define dso_local zeroext i1 @"?compareSingleFunctionMemptr@@
// X64:             (i8* %{{[^,]*}}, i8* %{{[^)]*}})
}

bool compareNeqSingleFunctionMemptr(void (Single::*l)(), void (Single::*r)()) {
  return l != r;
// Should only be one comparison here.
// CHECK: define dso_local zeroext i1 @"?compareNeqSingleFunctionMemptr@@YA_NP8Single@@AEXXZ0@Z"{{.*}} {
// CHECK-NOT: icmp
// CHECK:   %[[r:.*]] = icmp ne
// CHECK-NOT: icmp
// CHECK:   ret i1 %[[r]]
// CHECK: }
}

bool unspecFuncMemptrEq(void (Unspecified::*l)(), void (Unspecified::*r)()) {
  return l == r;
// CHECK: define dso_local zeroext i1 @"?unspecFuncMemptrEq@@YA_NP8Unspecified@@AEXXZ0@Z"{{.*}} {
// CHECK:   %[[lhs0:.*]] = extractvalue { i8*, i32, i32, i32 } %[[l:.*]], 0
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r:.*]], 0
// CHECK:   %[[cmp0:.*]] = icmp eq i8* %[[lhs0]], %{{.*}}
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 1
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 1
// CHECK:   %[[cmp1:.*]] = icmp eq i32
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 2
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 2
// CHECK:   %[[cmp2:.*]] = icmp eq i32
// CHECK:   %[[res12:.*]] = and i1 %[[cmp1]], %[[cmp2]]
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 3
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 3
// CHECK:   %[[cmp3:.*]] = icmp eq i32
// CHECK:   %[[res123:.*]] = and i1 %[[res12]], %[[cmp3]]
// CHECK:   %[[iszero:.*]] = icmp eq i8* %[[lhs0]], null
// CHECK:   %[[bits_or_null:.*]] = or i1 %[[res123]], %[[iszero]]
// CHECK:   %{{.*}} = and i1 %[[bits_or_null]], %[[cmp0]]
// CHECK:   ret i1 %{{.*}}
// CHECK: }

// X64-LABEL: define dso_local zeroext i1 @"?unspecFuncMemptrEq@@
// X64:             ({ i8*, i32, i32, i32 }* %0, { i8*, i32, i32, i32 }* %1)
}

bool unspecFuncMemptrNeq(void (Unspecified::*l)(), void (Unspecified::*r)()) {
  return l != r;
// CHECK: define dso_local zeroext i1 @"?unspecFuncMemptrNeq@@YA_NP8Unspecified@@AEXXZ0@Z"{{.*}} {
// CHECK:   %[[lhs0:.*]] = extractvalue { i8*, i32, i32, i32 } %[[l:.*]], 0
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r:.*]], 0
// CHECK:   %[[cmp0:.*]] = icmp ne i8* %[[lhs0]], %{{.*}}
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 1
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 1
// CHECK:   %[[cmp1:.*]] = icmp ne i32
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 2
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 2
// CHECK:   %[[cmp2:.*]] = icmp ne i32
// CHECK:   %[[res12:.*]] = or i1 %[[cmp1]], %[[cmp2]]
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[l]], 3
// CHECK:   %{{.*}} = extractvalue { i8*, i32, i32, i32 } %[[r]], 3
// CHECK:   %[[cmp3:.*]] = icmp ne i32
// CHECK:   %[[res123:.*]] = or i1 %[[res12]], %[[cmp3]]
// CHECK:   %[[iszero:.*]] = icmp ne i8* %[[lhs0]], null
// CHECK:   %[[bits_or_null:.*]] = and i1 %[[res123]], %[[iszero]]
// CHECK:   %{{.*}} = or i1 %[[bits_or_null]], %[[cmp0]]
// CHECK:   ret i1 %{{.*}}
// CHECK: }
}

bool unspecDataMemptrEq(int Unspecified::*l, int Unspecified::*r) {
  return l == r;
// CHECK: define dso_local zeroext i1 @"?unspecDataMemptrEq@@YA_NPQUnspecified@@H0@Z"{{.*}} {
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 0
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 0
// CHECK:   icmp eq i32
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 1
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 1
// CHECK:   icmp eq i32
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 2
// CHECK:   extractvalue { i32, i32, i32 } %{{.*}}, 2
// CHECK:   icmp eq i32
// CHECK:   and i1
// CHECK:   and i1
// CHECK:   ret i1
// CHECK: }

// X64-LABEL: define dso_local zeroext i1 @"?unspecDataMemptrEq@@
// X64:             ({ i32, i32, i32 }* %0, { i32, i32, i32 }* %1)
}

void (Multiple::*convertB2FuncToMultiple(void (B2::*mp)()))() {
  return mp;
// CHECK: define dso_local i64 @"?convertB2FuncToMultiple@@YAP8Multiple@@AEXXZP8B2@@AEXXZ@Z"{{.*}} {
// CHECK:   store
// CHECK:   %[[mp:.*]] = load i8*, i8** %{{.*}}, align 4
// CHECK:   icmp ne i8* %[[mp]], null
// CHECK:   br i1 %{{.*}} label %{{.*}}, label %{{.*}}
//
//        memptr.convert:                                   ; preds = %entry
// CHECK:   insertvalue { i8*, i32 } undef, i8* %[[mp]], 0
// CHECK:   insertvalue { i8*, i32 } %{{.*}}, i32 4, 1
// CHECK:   br label
//
//        memptr.converted:                                 ; preds = %memptr.convert, %entry
// CHECK:   phi { i8*, i32 } [ zeroinitializer, %{{.*}} ], [ {{.*}} ]
// CHECK: }
}

void (B2::*convertMultipleFuncToB2(void (Multiple::*mp)()))() {
// FIXME: cl emits warning C4407 on this code because of the representation
// change.  We might want to do the same.
  return static_cast<void (B2::*)()>(mp);
// FIXME: We should return i8* instead of i32 here.  The ptrtoint cast prevents
// LLVM from optimizing away the branch.  This is likely a bug in
// lib/CodeGen/TargetInfo.cpp with how we classify memptr types for returns.
//
// CHECK: define dso_local i32 @"?convertMultipleFuncToB2@@YAP8B2@@AEXXZP8Multiple@@AEXXZ@Z"{{.*}} {
// CHECK:   store
// CHECK:   %[[src:.*]] = load { i8*, i32 }, { i8*, i32 }* %{{.*}}, align 4
// CHECK:   extractvalue { i8*, i32 } %[[src]], 0
// CHECK:   icmp ne i8* %{{.*}}, null
// CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
//
//        memptr.convert:                                   ; preds = %entry
// CHECK:   %[[fp:.*]] = extractvalue { i8*, i32 } %[[src]], 0
// CHECK:   br label
//
//        memptr.converted:                                 ; preds = %memptr.convert, %entry
// CHECK:   phi i8* [ null, %{{.*}} ], [ %[[fp]], %{{.*}} ]
// CHECK: }
}

namespace Test1 {

struct A { int a; };
struct B { int b; };
struct C : virtual A { int c; };
struct D : B, C { int d; };

void (D::*convertCToD(void (C::*mp)()))() {
  return mp;
// CHECK: define dso_local void @"?convertCToD@Test1@@YAP8D@1@AEXXZP8C@1@AEXXZ@Z"{{.*}} {
// CHECK:   store
// CHECK:   load { i8*, i32, i32 }, { i8*, i32, i32 }* %{{.*}}, align 4
// CHECK:   extractvalue { i8*, i32, i32 } %{{.*}}, 0
// CHECK:   icmp ne i8* %{{.*}}, null
// CHECK:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
//
//        memptr.convert:                                   ; preds = %entry
// CHECK:   extractvalue { i8*, i32, i32 } %{{.*}}, 0
// CHECK:   %[[nvoff:.*]] = extractvalue { i8*, i32, i32 } %{{.*}}, 1
// CHECK:   %[[vbidx:.*]] = extractvalue { i8*, i32, i32 } %{{.*}}, 2
// CHECK:   %[[is_nvbase:.*]] = icmp eq i32 %[[vbidx]], 0
// CHECK:   %[[nv_disp:.*]] = add nsw i32 %[[nvoff]], 4
// CHECK:   %[[nv_adj:.*]] = select i1 %[[is_nvbase]], i32 %[[nv_disp]], i32 0
// CHECK:   %[[dst_adj:.*]] = select i1 %[[is_nvbase]], i32 4, i32 0
// CHECK:   %[[adj:.*]] = sub nsw i32 %[[nv_adj]], %[[dst_adj]]
// CHECK:   insertvalue { i8*, i32, i32 } undef, i8* {{.*}}, 0
// CHECK:   insertvalue { i8*, i32, i32 } {{.*}}, i32 %[[adj]], 1
// CHECK:   insertvalue { i8*, i32, i32 } {{.*}}, i32 {{.*}}, 2
// CHECK:   br label
//
//        memptr.converted:                                 ; preds = %memptr.convert, %entry
// CHECK:   phi { i8*, i32, i32 } [ { i8* null, i32 0, i32 -1 }, {{.*}} ], [ {{.*}} ]
// CHECK: }
}

}

namespace Test2 {
// Test that we dynamically convert between different null reps.

struct A { int a; };
struct B : A { int b; };
struct C : A {
  int c;
  virtual void hasVfPtr();
};

int A::*reinterpret(int B::*mp) {
  return reinterpret_cast<int A::*>(mp);
// CHECK: define dso_local i32 @"?reinterpret@Test2@@YAPQA@1@HPQB@1@H@Z"{{.*}}  {
// CHECK-NOT: select
// CHECK:   ret i32
// CHECK: }
}

int A::*reinterpret(int C::*mp) {
  return reinterpret_cast<int A::*>(mp);
// CHECK: define dso_local i32 @"?reinterpret@Test2@@YAPQA@1@HPQC@1@H@Z"{{.*}}  {
// CHECK:   %[[mp:.*]] = load i32, i32*
// CHECK:   %[[cmp:.*]] = icmp ne i32 %[[mp]], 0
// CHECK:   select i1 %[[cmp]], i32 %[[mp]], i32 -1
// CHECK: }
}

}

namespace Test3 {
// Make sure we cast 'this' to i8* before using GEP.

struct A {
  int a;
  int b;
};

int *load_data(A *a, int A::*mp) {
  return &(a->*mp);
// CHECK-LABEL: define dso_local i32* @"?load_data@Test3@@YAPAHPAUA@1@PQ21@H@Z"{{.*}}  {
// CHECK:    %[[a:.*]] = load %"struct.Test3::A"*, %"struct.Test3::A"** %{{.*}}, align 4
// CHECK:    %[[mp:.*]] = load i32, i32* %{{.*}}, align 4
// CHECK:    %[[a_i8:.*]] = bitcast %"struct.Test3::A"* %[[a]] to i8*
// CHECK:    getelementptr inbounds i8, i8* %[[a_i8]], i32 %[[mp]]
// CHECK: }
}

}

namespace Test4 {

struct A        { virtual void f(); };
struct B        { virtual void g(); };
struct C : A, B { virtual void g(); };

void (C::*getmp())() {
  return &C::g;
}
// CHECK-LABEL: define dso_local i64 @"?getmp@Test4@@YAP8C@1@AEXXZXZ"()
// CHECK: store { i8*, i32 } { i8* bitcast (void (%"struct.Test4::C"*, ...)* @"??_9C@Test4@@$BA@AE" to i8*), i32 4 }, { i8*, i32 }* %{{.*}}
//

// CHECK-LABEL: define linkonce_odr x86_thiscallcc void @"??_9C@Test4@@$BA@AE"(%"struct.Test4::C"* %this, ...) {{.*}} comdat
// CHECK-NOT:  getelementptr
// CHECK:  load void (%"struct.Test4::C"*, ...)**, void (%"struct.Test4::C"*, ...)*** %{{.*}}
// CHECK:  getelementptr inbounds void (%"struct.Test4::C"*, ...)*, void (%"struct.Test4::C"*, ...)** %{{.*}}, i64 0
// CHECK-NOT:  getelementptr
// CHECK:  musttail call x86_thiscallcc void (%"struct.Test4::C"*, ...) %

}

namespace pr20007 {
struct A {
  void f();
  void f(int);
};
struct B : public A {};
void test() { void (B::*a)() = &B::f; }
// CHECK-LABEL: define dso_local void @"?test@pr20007@@YAXXZ"
// CHECK: store i8* bitcast (void (%"struct.pr20007::A"*)* @"?f@A@pr20007@@QAEXXZ" to i8*)
}

namespace pr20007_kw {
struct A {
  void f();
  void f(int);
};
struct __single_inheritance B;
struct B : public A {};
void test() { void (B::*a)() = &B::f; }
// CHECK-LABEL: define dso_local void @"?test@pr20007_kw@@YAXXZ"
// CHECK: store i8* bitcast (void (%"struct.pr20007_kw::A"*)* @"?f@A@pr20007_kw@@QAEXXZ" to i8*)
}

namespace pr20007_pragma {
struct A {
  void f();
  void f(int);
};
struct B : public A {};
void test() { (void)(void (B::*)()) &B::f; }
#pragma pointers_to_members(full_generality, virtual_inheritance)
static_assert(sizeof(int B::*) == 4, "");
static_assert(sizeof(int A::*) == 4, "");
#pragma pointers_to_members(best_case)
// CHECK-LABEL: define dso_local void @"?test@pr20007_pragma@@YAXXZ"
}

namespace pr20007_pragma2 {
struct A {
};
struct B : public A {
  void f();
};
void test() { (void)&B::f; }
#pragma pointers_to_members(full_generality, virtual_inheritance)
static_assert(sizeof(int B::*) == 4, "");
static_assert(sizeof(int A::*) == 12, "");
#pragma pointers_to_members(best_case)
// CHECK-LABEL: define dso_local void @"?test@pr20007_pragma2@@YAXXZ"
}

namespace pr23823 {
struct Base { void Method(); };
struct Child : Base {};
void use(void (Child::*const &)());
void f() { use(&Child::Method); }
#pragma pointers_to_members(full_generality, virtual_inheritance)
static_assert(sizeof(int Base::*) == 4, "");
static_assert(sizeof(int Child::*) == 4, "");
#pragma pointers_to_members(best_case)
}

namespace pr19987 {
template <typename T>
struct S {
  int T::*x;
};

struct U : S<U> {};

static_assert(sizeof(S<U>::x) == 12, "");
}

#else
struct __virtual_inheritance A;
#ifdef MEMFUN
int foo(A *a, int (A::*mp)()) {
    return (a->*mp)(); // expected-error{{requires a complete class type}}
}
#else
int foo(A *a, int A::*mp) {
    return a->*mp; // expected-error{{requires a complete class type}}
}
#endif
#endif

namespace pr23878 {
struct A { virtual void g(); };
struct B { virtual void f(); };
struct C : virtual B { void f(); };
struct D : A, C {};

typedef void (D::*DMemPtrTy)();

// CHECK-LABEL: define dso_local void @"?get_memptr@pr23878@@YAP8D@1@AEXXZXZ"
// CHECK: @"??_9C@pr23878@@$BA@AE" to i8*), i32 0, i32 4
DMemPtrTy get_memptr() { return &D::f; }
}

class C {};

typedef void (C::*f)();

class CA : public C {
public:
  void OnHelp(void);
  int OnHelp(int);
};

// CHECK-LABEL: foo_fun
void foo_fun() {
  // CHECK: store i8* bitcast (void (%class.CA*)* @"?OnHelp@CA@@QAEXXZ" to i8*), i8**
  f func = (f)&CA::OnHelp;
}
namespace PR24703 {
struct S;

void f(int S::*&p) {}
// CHECK-LABEL: define dso_local void @"?f@PR24703@@YAXAAPQS@1@H@Z"(
}

namespace ReferenceToMPTWithIncompleteClass {
struct S;
struct J;
struct K;
extern K *k;

// CHECK-LABEL: @"?f@ReferenceToMPTWithIncompleteClass@@YAIAAPQS@1@H@Z"(
// CHECK: ret i32 12
unsigned f(int S::*&p) { return sizeof p; }

// CHECK-LABEL: @"?g@ReferenceToMPTWithIncompleteClass@@YA_NAAPQJ@1@H0@Z"(
bool g(int J::*&p, int J::*&q) { return p == q; }

// CHECK-LABEL: @"?h@ReferenceToMPTWithIncompleteClass@@YAHAAPQK@1@H@Z"(
int h(int K::*&p) { return k->*p; }
}

namespace PMFInTemplateArgument {
template <class C, int (C::*M)(int)>
void JSMethod();
class A {
  int printd(int);
  void printd();
};
void A::printd() { JSMethod<A, &A::printd>(); }
// CHECK-LABEL: @"??$JSMethod@VA@PMFInTemplateArgument@@$1?printd@12@AAEHH@Z@PMFInTemplateArgument@@YAXXZ"(
}
