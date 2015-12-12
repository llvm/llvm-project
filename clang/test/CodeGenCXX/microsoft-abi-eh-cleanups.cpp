// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=i386-pc-win32 -mconstructor-aliases -fexceptions -fcxx-exceptions -fno-rtti | FileCheck -check-prefix WIN32 %s

struct A {
  A();
  ~A();
  int a;
};

A getA();

int TakesTwo(A a, A b);
void HasEHCleanup() {
  TakesTwo(getA(), getA());
}

// With exceptions, we need to clean up at least one of these temporaries.
// WIN32-LABEL: define void @"\01?HasEHCleanup@@YAXXZ"() {{.*}} {
// WIN32:   %[[base:.*]] = call i8* @llvm.stacksave()
//    If this call throws, we have to restore the stack.
// WIN32:   call void @"\01?getA@@YA?AUA@@XZ"(%struct.A* sret %{{.*}})
//    If this call throws, we have to cleanup the first temporary.
// WIN32:   invoke void @"\01?getA@@YA?AUA@@XZ"(%struct.A* sret %{{.*}})
//    If this call throws, we have to cleanup the stacksave.
// WIN32:   call i32 @"\01?TakesTwo@@YAHUA@@0@Z"
// WIN32:   call void @llvm.stackrestore
// WIN32:   ret void
//
//    There should be one dtor call for unwinding from the second getA.
// WIN32:   cleanuppad
// WIN32:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"({{.*}})
// WIN32-NOT: @"\01??1A@@QAE@XZ"
// WIN32: }

void TakeRef(const A &a);
int HasDeactivatedCleanups() {
  return TakesTwo((TakeRef(A()), A()), (TakeRef(A()), A()));
}

// WIN32-LABEL: define i32 @"\01?HasDeactivatedCleanups@@YAHXZ"() {{.*}} {
// WIN32:   %[[isactive:.*]] = alloca i1
// WIN32:   call i8* @llvm.stacksave()
// WIN32:   %[[argmem:.*]] = alloca inalloca [[argmem_ty:<{ %struct.A, %struct.A }>]]
// WIN32:   %[[arg1:.*]] = getelementptr inbounds [[argmem_ty]], [[argmem_ty]]* %[[argmem]], i32 0, i32 1
// WIN32:   call x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   invoke void @"\01?TakeRef@@YAXABUA@@@Z"
//
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"(%struct.A* %[[arg1]])
// WIN32:   store i1 true, i1* %[[isactive]]
//
// WIN32:   %[[arg0:.*]] = getelementptr inbounds [[argmem_ty]], [[argmem_ty]]* %[[argmem]], i32 0, i32 0
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   invoke void @"\01?TakeRef@@YAXABUA@@@Z"
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   store i1 false, i1* %[[isactive]]
//
// WIN32:   invoke i32 @"\01?TakesTwo@@YAHUA@@0@Z"([[argmem_ty]]* inalloca %[[argmem]])
//        Destroy the two const ref temporaries.
// WIN32:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"({{.*}})
// WIN32:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"({{.*}})
// WIN32:   ret i32
//
//        Conditionally destroy arg1.
// WIN32:   %[[cond:.*]] = load i1, i1* %[[isactive]]
// WIN32:   br i1 %[[cond]]
// WIN32:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"(%struct.A* %[[arg1]])
// WIN32: }

// Test putting the cleanups inside a conditional.
int CouldThrow();
int HasConditionalCleanup(bool cond) {
  return (cond ? TakesTwo(A(), A()) : CouldThrow());
}

// WIN32-LABEL: define i32 @"\01?HasConditionalCleanup@@YAH_N@Z"(i1 zeroext %{{.*}}) {{.*}} {
// WIN32:   store i1 false
// WIN32:   br i1
// WIN32:   call i8* @llvm.stacksave()
// WIN32:   call x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"(%struct.A* %{{.*}})
// WIN32:   store i1 true
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"(%struct.A* %{{.*}})
// WIN32:   call i32 @"\01?TakesTwo@@YAHUA@@0@Z"
//
// WIN32:   call void @llvm.stackrestore
//
// WIN32:   call i32 @"\01?CouldThrow@@YAHXZ"()
//
//        Only one dtor in the invoke for arg1
// WIN32:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"({{.*}})
// WIN32-NOT: invoke x86_thiscallcc void @"\01??1A@@QAE@XZ"
// WIN32: }

// Now test both.
int HasConditionalDeactivatedCleanups(bool cond) {
  return (cond ? TakesTwo((TakeRef(A()), A()), (TakeRef(A()), A())) : CouldThrow());
}

// WIN32-LABEL: define i32 @"\01?HasConditionalDeactivatedCleanups@@YAH_N@Z"{{.*}} {
// WIN32:   alloca i1
// WIN32:   %[[arg1_cond:.*]] = alloca i1
//        Start all four cleanups as deactivated.
// WIN32:   store i1 false
// WIN32:   store i1 false
// WIN32:   store i1 false
// WIN32:   store i1 false
// WIN32:   br i1
//        True condition.
// WIN32:   call x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   store i1 true
// WIN32:   invoke void @"\01?TakeRef@@YAXABUA@@@Z"
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   store i1 true, i1* %[[arg1_cond]]
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   store i1 true
// WIN32:   invoke void @"\01?TakeRef@@YAXABUA@@@Z"
// WIN32:   invoke x86_thiscallcc %struct.A* @"\01??0A@@QAE@XZ"
// WIN32:   store i1 true
// WIN32:   store i1 false, i1* %[[arg1_cond]]
// WIN32:   invoke i32 @"\01?TakesTwo@@YAHUA@@0@Z"
//        False condition.
// WIN32:   invoke i32 @"\01?CouldThrow@@YAHXZ"()
//        Two normal cleanups for TakeRef args.
// WIN32:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"({{.*}})
// WIN32-NOT:   invoke x86_thiscallcc void @"\01??1A@@QAE@XZ"
// WIN32:   ret i32
//
//        Somewhere in the landing pad soup, we conditionally destroy arg1.
// WIN32:   %[[isactive:.*]] = load i1, i1* %[[arg1_cond]]
// WIN32:   br i1 %[[isactive]]
// WIN32:   call x86_thiscallcc void @"\01??1A@@QAE@XZ"({{.*}})
// WIN32: }

namespace crash_on_partial_destroy {
struct A {
  virtual ~A();
};

struct B : virtual A {
  // Has an implicit destructor.
};

struct C : B {
  C();
};

void foo();
// We used to crash when emitting this.
C::C() { foo(); }

// Verify that we don't bother with a vbtable lookup when adjusting the this
// pointer to call a base destructor from a constructor while unwinding.
// WIN32-LABEL: define {{.*}} @"\01??0C@crash_on_partial_destroy@@QAE@XZ"{{.*}} {
// WIN32:      cleanuppad
//
//        We shouldn't do any vbptr loads, just constant GEPs.
// WIN32-NOT:  load
// WIN32:      getelementptr i8, i8* %{{.*}}, i32 4
// WIN32-NOT:  load
// WIN32:      bitcast i8* %{{.*}} to %"struct.crash_on_partial_destroy::B"*
// WIN32:      call x86_thiscallcc void @"\01??1B@crash_on_partial_destroy@@UAE@XZ"
//
// WIN32-NOT:  load
// WIN32:      bitcast %"struct.crash_on_partial_destroy::C"* %{{.*}} to i8*
// WIN32-NOT:  load
// WIN32:      getelementptr inbounds i8, i8* %{{.*}}, i32 4
// WIN32-NOT:  load
// WIN32:      bitcast i8* %{{.*}} to %"struct.crash_on_partial_destroy::A"*
// WIN32:      call x86_thiscallcc void @"\01??1A@crash_on_partial_destroy@@UAE@XZ"({{.*}})
// WIN32: }
}

namespace dont_call_terminate {
struct C {
  ~C();
};
void g();
void f() {
  C c;
  g();
}

// WIN32-LABEL: define void @"\01?f@dont_call_terminate@@YAXXZ"()
// WIN32: invoke void @"\01?g@dont_call_terminate@@YAXXZ"()
// WIN32-NEXT: to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// WIN32: [[cont]]
// WIN32: call x86_thiscallcc void @"\01??1C@dont_call_terminate@@QAE@XZ"({{.*}})
//
// WIN32: [[lpad]]
// WIN32-NEXT: cleanuppad
// WIN32: call x86_thiscallcc void @"\01??1C@dont_call_terminate@@QAE@XZ"({{.*}})
}

namespace noexcept_false_dtor {
struct D {
  ~D() noexcept(false);
};
void f() {
  D d;
  CouldThrow();
}
}

// WIN32-LABEL: define void @"\01?f@noexcept_false_dtor@@YAXXZ"()
// WIN32: invoke i32 @"\01?CouldThrow@@YAHXZ"()
// WIN32: call x86_thiscallcc void @"\01??1D@noexcept_false_dtor@@QAE@XZ"(%"struct.noexcept_false_dtor::D"* %{{.*}})
// WIN32: cleanuppad
// WIN32: call x86_thiscallcc void @"\01??1D@noexcept_false_dtor@@QAE@XZ"(%"struct.noexcept_false_dtor::D"* %{{.*}})
// WIN32: cleanupret
