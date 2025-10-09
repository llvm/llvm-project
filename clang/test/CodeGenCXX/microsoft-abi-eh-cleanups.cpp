// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - -triple=i386-pc-win32 -mconstructor-aliases -fexceptions -fcxx-exceptions -fno-rtti | FileCheck -check-prefix WIN32 -check-prefix WIN32-O0 %s
// RUN: %clang_cc1 -std=c++11 -emit-llvm -O3 -disable-llvm-passes %s -o - -triple=i386-pc-win32 -mconstructor-aliases -fexceptions -fcxx-exceptions -fno-rtti | FileCheck -check-prefix WIN32 -check-prefix WIN32-O3 -check-prefix WIN32-LIFETIME %s

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
// WIN32-LABEL: define dso_local void @"?HasEHCleanup@@YAXXZ"() {{.*}} {
// WIN32:   %[[base:.*]] = call ptr @llvm.stacksave.p0()
//    If this call throws, we have to restore the stack.
// WIN32:   call void @"?getA@@YA?AUA@@XZ"(ptr dead_on_unwind writable sret(%struct.A) align 4 %{{.*}})
//    If this call throws, we have to cleanup the first temporary.
// WIN32:   invoke void @"?getA@@YA?AUA@@XZ"(ptr dead_on_unwind writable sret(%struct.A) align 4 %{{.*}})
//    If this call throws, we have to cleanup the stacksave.
// WIN32:   call noundef i32 @"?TakesTwo@@YAHUA@@0@Z"
// WIN32:   call void @llvm.stackrestore
// WIN32:   ret void
//
//    There should be one dtor call for unwinding from the second getA.
// WIN32:   cleanuppad
// WIN32:   call x86_thiscallcc void @"??1A@@QAE@XZ"({{.*}})
// WIN32-NOT: @"??1A@@QAE@XZ"
// WIN32: }


// This test verifies the fix for a crash that occurred after
// fa87fa97fb79.
void HasEHCleanupNoexcept() noexcept {
  TakesTwo(getA(), getA());
}

// With exceptions, we need to clean up at least one of these temporaries.
// WIN32-LABEL: define dso_local void @"?HasEHCleanupNoexcept@@YAXXZ"() {{.*}} {
// WIN32:   %[[base:.*]] = call ptr @llvm.stacksave.p0()
// WIN32:   invoke void @"?getA@@YA?AUA@@XZ"(ptr dead_on_unwind writable sret(%struct.A) align 4 %{{.*}})
// WIN32:   invoke void @"?getA@@YA?AUA@@XZ"(ptr dead_on_unwind writable sret(%struct.A) align 4 %{{.*}})
// WIN32:   invoke noundef i32 @"?TakesTwo@@YAHUA@@0@Z"
// WIN32:   call void @llvm.stackrestore
// WIN32:   ret void
//
//    Since all the calls terminate, there should be no dtors on the unwind
// WIN32:   cleanuppad
// WIN32-NOT: @"??1A@@QAE@XZ"
// WIN32: }


void TakeRef(const A &a);
int HasDeactivatedCleanups() {
  return TakesTwo((TakeRef(A()), A()), (TakeRef(A()), A()));
}

// WIN32-LABEL: define dso_local noundef i32 @"?HasDeactivatedCleanups@@YAHXZ"() {{.*}} {
// WIN32:   %[[isactive:.*]] = alloca i1
// WIN32:   call ptr @llvm.stacksave.p0()
// WIN32:   %[[argmem:.*]] = alloca inalloca [[argmem_ty:<{ %struct.A, %struct.A }>]]
// WIN32:   %[[arg1:.*]] = getelementptr inbounds nuw [[argmem_ty]], ptr %[[argmem]], i32 0, i32 1
// WIN32:   call x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"
// WIN32:   invoke void @"?TakeRef@@YAXABUA@@@Z"
//
// WIN32:   invoke x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"(ptr {{[^,]*}} %[[arg1]])
// WIN32:   store i1 true, ptr %[[isactive]]
//
// WIN32:   %[[arg0:.*]] = getelementptr inbounds nuw [[argmem_ty]], ptr %[[argmem]], i32 0, i32 0
// WIN32:   invoke x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"
// WIN32:   invoke void @"?TakeRef@@YAXABUA@@@Z"
// WIN32:   invoke x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"
// WIN32:   store i1 false, ptr %[[isactive]]
//
// WIN32:   invoke noundef i32 @"?TakesTwo@@YAHUA@@0@Z"(ptr inalloca([[argmem_ty]]) %[[argmem]])
//        Destroy the two const ref temporaries.
// WIN32:   call x86_thiscallcc void @"??1A@@QAE@XZ"({{.*}})
// WIN32:   call x86_thiscallcc void @"??1A@@QAE@XZ"({{.*}})
// WIN32:   ret i32
//
//        Conditionally destroy arg1.
// WIN32:   %[[cond:.*]] = load i1, ptr %[[isactive]]
// WIN32:   br i1 %[[cond]]
// WIN32:   call x86_thiscallcc void @"??1A@@QAE@XZ"(ptr {{[^,]*}} %[[arg1]])
// WIN32: }

// Test putting the cleanups inside a conditional.
int CouldThrow();
int HasConditionalCleanup(bool cond) {
  return (cond ? TakesTwo(A(), A()) : CouldThrow());
}

// WIN32-LABEL: define dso_local noundef i32 @"?HasConditionalCleanup@@YAH_N@Z"(i1 noundef zeroext %{{.*}}) {{.*}} {
// WIN32:   store i1 false
// WIN32:   br i1
// WIN32:   call ptr @llvm.stacksave.p0()
// WIN32:   call x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
// WIN32:   store i1 true
// WIN32:   invoke x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
// WIN32:   call noundef i32 @"?TakesTwo@@YAHUA@@0@Z"
//
// WIN32:   call void @llvm.stackrestore
//
// WIN32:   call noundef i32 @"?CouldThrow@@YAHXZ"()
//
//        Only one dtor in the invoke for arg1
// WIN32:   call x86_thiscallcc void @"??1A@@QAE@XZ"({{.*}})
// WIN32-NOT: invoke x86_thiscallcc void @"??1A@@QAE@XZ"
// WIN32: }

// Now test both.
int HasConditionalDeactivatedCleanups(bool cond) {
  return (cond ? TakesTwo((TakeRef(A()), A()), (TakeRef(A()), A())) : CouldThrow());
}

// WIN32-O0-LABEL: define dso_local noundef i32 @"?HasConditionalDeactivatedCleanups@@YAH_N@Z"{{.*}} {
// WIN32-O0:   alloca i1
// WIN32-O0:   %[[arg1_cond:.*]] = alloca i1
//        Start all four cleanups as deactivated.
// WIN32-O0:   store i1 false
// WIN32-O0:   store i1 false
// WIN32-O0:   store i1 false
// WIN32-O0:   store i1 false
// WIN32-O0:   br i1
//        True condition.
// WIN32-O0:   call x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"
// WIN32-O0:   store i1 true
// WIN32-O0:   invoke void @"?TakeRef@@YAXABUA@@@Z"
// WIN32-O0:   invoke x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"
// WIN32-O0:   store i1 true, ptr %[[arg1_cond]]
// WIN32-O0:   invoke x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"
// WIN32-O0:   store i1 true
// WIN32-O0:   invoke void @"?TakeRef@@YAXABUA@@@Z"
// WIN32-O0:   invoke x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"
// WIN32-O0:   store i1 true
// WIN32-O0:   store i1 false, ptr %[[arg1_cond]]
// WIN32-O0:   invoke noundef i32 @"?TakesTwo@@YAHUA@@0@Z"
//        False condition.
// WIN32-O0:   invoke noundef i32 @"?CouldThrow@@YAHXZ"()
//        Two normal cleanups for TakeRef args.
// WIN32-O0:   call x86_thiscallcc void @"??1A@@QAE@XZ"({{.*}})
// WIN32-O0-NOT:   invoke x86_thiscallcc void @"??1A@@QAE@XZ"
// WIN32-O0:   ret i32
//
//        Somewhere in the landing pad soup, we conditionally destroy arg1.
// WIN32-O0:   %[[isactive:.*]] = load i1, ptr %[[arg1_cond]]
// WIN32-O0:   br i1 %[[isactive]]
// WIN32-O0:   call x86_thiscallcc void @"??1A@@QAE@XZ"({{.*}})
// WIN32-O0: }

// WIN32-O3-LABEL: define dso_local noundef i32 @"?HasConditionalDeactivatedCleanups@@YAH_N@Z"{{.*}} {
// WIN32-O3:   alloca i1
// WIN32-O3:   alloca i1
// WIN32-O3:   %[[arg1_cond:.*]] = alloca i1
//        Start all four cleanups as deactivated.
// WIN32-O3:   store i1 false
// WIN32-O3:   store i1 false
// WIN32-O3:   store i1 false
// WIN32-O3:   store i1 false
// WIN32-O3:   store i1 false
// WIN32-O3:   store i1 false
// WIN32-O3:   br i1
//        True condition.
// WIN32-O3:   call x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"
// WIN32-O3:   store i1 true
// WIN32-O3:   invoke void @"?TakeRef@@YAXABUA@@@Z"
// WIN32-O3:   invoke x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"
// WIN32-O3:   store i1 true, ptr %[[arg1_cond]]
// WIN32-O3:   invoke x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"
// WIN32-O3:   store i1 true
// WIN32-O3:   invoke void @"?TakeRef@@YAXABUA@@@Z"
// WIN32-O3:   invoke x86_thiscallcc noundef ptr @"??0A@@QAE@XZ"
// WIN32-O3:   store i1 true
// WIN32-O3:   store i1 false, ptr %[[arg1_cond]]
// WIN32-O3:   invoke noundef i32 @"?TakesTwo@@YAHUA@@0@Z"
//        False condition.
// WIN32-O3:   invoke noundef i32 @"?CouldThrow@@YAHXZ"()
//        Two normal cleanups for TakeRef args.
// WIN32-O3:   call x86_thiscallcc void @"??1A@@QAE@XZ"({{.*}})
// WIN32-O3-NOT:   invoke x86_thiscallcc void @"??1A@@QAE@XZ"
// WIN32-O3:   ret i32
//
//        Somewhere in the landing pad soup, we conditionally destroy arg1.
// WIN32-O3:   %[[isactive:.*]] = load i1, ptr %[[arg1_cond]]
// WIN32-O3:   br i1 %[[isactive]]
// WIN32-O3:   call x86_thiscallcc void @"??1A@@QAE@XZ"({{.*}})
// WIN32-O3: }

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
// WIN32-LABEL: define dso_local {{.*}} @"??0C@crash_on_partial_destroy@@QAE@XZ"{{.*}} {
// WIN32:      cleanuppad
//
//        We shouldn't do any vbptr loads, just constant GEPs.
// WIN32-NOT:  load
// WIN32:      getelementptr i8, ptr %{{.*}}, i32 4
// WIN32-NOT:  load
// WIN32:      call x86_thiscallcc void @"??1B@crash_on_partial_destroy@@UAE@XZ"
//
// WIN32-NOT:  load
// WIN32-NOT:  load
// WIN32:      getelementptr inbounds i8, ptr %{{.*}}, i32 4
// WIN32-NOT:  load
// WIN32:      call x86_thiscallcc void @"??1A@crash_on_partial_destroy@@UAE@XZ"({{.*}})
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

// WIN32-LABEL: define dso_local void @"?f@dont_call_terminate@@YAXXZ"()
// WIN32: invoke void @"?g@dont_call_terminate@@YAXXZ"()
// WIN32-NEXT: to label %[[cont:[^ ]*]] unwind label %[[lpad:[^ ]*]]
//
// WIN32: [[cont]]
// WIN32: call x86_thiscallcc void @"??1C@dont_call_terminate@@QAE@XZ"({{.*}})
//
// WIN32: [[lpad]]
// WIN32-NEXT: cleanuppad
// WIN32: call x86_thiscallcc void @"??1C@dont_call_terminate@@QAE@XZ"({{.*}})
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

// WIN32-LABEL: define dso_local void @"?f@noexcept_false_dtor@@YAXXZ"()
// WIN32: invoke noundef i32 @"?CouldThrow@@YAHXZ"()
// WIN32: call x86_thiscallcc void @"??1D@noexcept_false_dtor@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
// WIN32: cleanuppad
// WIN32: call x86_thiscallcc void @"??1D@noexcept_false_dtor@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})
// WIN32: cleanupret

namespace lifetime_marker {
struct C {
  ~C();
};
void g();
void f() {
  C c;
  g();
}

// WIN32-LIFETIME-LABEL: define dso_local void @"?f@lifetime_marker@@YAXXZ"()
// WIN32-LIFETIME: %[[c:.*]] = alloca %"struct.lifetime_marker::C"
// WIN32-LIFETIME: call void @llvm.lifetime.start.p0(ptr %c)
// WIN32-LIFETIME: invoke void @"?g@lifetime_marker@@YAXXZ"()
// WIN32-LIFETIME-NEXT: to label %[[cont:[^ ]*]] unwind label %[[lpad0:[^ ]*]]
//
// WIN32-LIFETIME: [[cont]]
// WIN32-LIFETIME: call x86_thiscallcc void @"??1C@lifetime_marker@@QAE@XZ"({{.*}})
// WIN32-LIFETIME: call void @llvm.lifetime.end.p0(ptr %[[c]])
//
// WIN32-LIFETIME: [[lpad0]]
// WIN32-LIFETIME-NEXT: cleanuppad
// WIN32-LIFETIME: call x86_thiscallcc void @"??1C@lifetime_marker@@QAE@XZ"({{.*}})
// WIN32-LIFETIME: cleanupret {{.*}} unwind label %[[lpad1:[^ ]*]]
//
// WIN32-LIFETIME: [[lpad1]]
// WIN32-LIFETIME-NEXT: cleanuppad
// WIN32-LIFETIME: call void @llvm.lifetime.end.p0(ptr %[[c]])
}

struct class_2 {
  class_2();
  virtual ~class_2();
};
struct class_1 : virtual class_2 {
  class_1(){throw "Unhandled exception";}
  virtual ~class_1() {}
};
struct class_0 : class_1 {
  class_0() ;
  virtual ~class_0() {}
};

class_0::class_0() {
  // WIN32: define dso_local x86_thiscallcc noundef ptr @"??0class_0@@QAE@XZ"(ptr {{[^,]*}} returned align 4 dereferenceable(4) %this, i32 noundef %is_most_derived)
  // WIN32: store i32 %is_most_derived, ptr %[[IS_MOST_DERIVED_VAR:.*]], align 4
  // WIN32: %[[IS_MOST_DERIVED_VAL:.*]] = load i32, ptr %[[IS_MOST_DERIVED_VAR]]
  // WIN32: %[[SHOULD_CALL_VBASE_CTORS:.*]] = icmp ne i32 %[[IS_MOST_DERIVED_VAL]], 0
  // WIN32: br i1 %[[SHOULD_CALL_VBASE_CTORS]], label %[[INIT_VBASES:.*]], label %[[SKIP_VBASES:.*]]
  // WIN32: [[INIT_VBASES]]
  // WIN32: br label %[[SKIP_VBASES]]
  // WIN32: [[SKIP_VBASES]]
// ehcleanup:
  // WIN32: %[[CLEANUPPAD:.*]] = cleanuppad within none []
  // WIN32-NEXT: getelementptr inbounds i8, ptr %{{.*}}, i{{.*}} {{.}}
  // WIN32-NEXT: %[[SHOULD_CALL_VBASE_DTOR:.*]] = icmp ne i32 %[[IS_MOST_DERIVED_VAL]], 0
  // WIN32-NEXT: br i1 %[[SHOULD_CALL_VBASE_DTOR]], label %[[DTOR_VBASE:.*]], label %[[SKIP_VBASE:.*]]
  // WIN32: [[DTOR_VBASE]]
  // WIN32-NEXT: call x86_thiscallcc void @"??1class_2@@UAE@XZ"
  // WIN32: br label %[[SKIP_VBASE]]
  // WIN32: [[SKIP_VBASE]]
}

namespace PR37146 {
// Check that IRGen doesn't emit calls to synthesized destructors for
// non-trival C structs.

// WIN32: define dso_local void @"?test@PR37146@@YAXXZ"()
// WIN32: call void @llvm.memset.p0.i32(
// WIN32: call i32 @"?getS@PR37146@@YA?AUS@1@XZ"(
// WIN32: call void @"?func@PR37146@@YAXUS@1@0@Z"(
// WIN32-NEXT: ret void
// WIN32-NEXT: {{^}$}}

struct S {
  int f;
};

void func(S, S);
S getS();

void test() {
  func(getS(), S());
}

}
