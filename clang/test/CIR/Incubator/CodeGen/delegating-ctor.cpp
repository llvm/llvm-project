// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fexceptions -fcxx-exceptions %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t-og.ll %s

struct Delegating {
  Delegating();
  Delegating(int);
};

// Check that the constructor being delegated to is called with the correct
// arguments.
Delegating::Delegating() : Delegating(0) {}

// CHECK-LABEL: cir.func {{.*}} @_ZN10DelegatingC2Ev(%arg0: !cir.ptr<!rec_Delegating> {{.*}}) {{.*}} {
// CHECK-NEXT:    %0 = cir.alloca !cir.ptr<!rec_Delegating>, !cir.ptr<!cir.ptr<!rec_Delegating>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:    cir.store{{.*}} %arg0, %0 : !cir.ptr<!rec_Delegating>, !cir.ptr<!cir.ptr<!rec_Delegating>>
// CHECK-NEXT:    %1 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_Delegating>>, !cir.ptr<!rec_Delegating>
// CHECK-NEXT:    %2 = cir.const #cir.int<0> : !s32i
// CHECK-NEXT:    cir.call @_ZN10DelegatingC2Ei(%1, %2) : (!cir.ptr<!rec_Delegating>, !s32i) -> ()
// CHECK-NEXT:    cir.return
// CHECK-NEXT:  }

struct DelegatingWithZeroing {
  int i;
  DelegatingWithZeroing() = default;
  DelegatingWithZeroing(int);
};

// Check that the delegating constructor performs zero-initialization here.
// The trivial default constructor call is now lowered away as expected.
DelegatingWithZeroing::DelegatingWithZeroing(int) : DelegatingWithZeroing() {}

// CHECK-LABEL: cir.func {{.*}} @_ZN21DelegatingWithZeroingC2Ei(%arg0: !cir.ptr<!rec_DelegatingWithZeroing> {{.*}}, %arg1: !s32i {{.*}}) {{.*}} {
// CHECK-NEXT:    %0 = cir.alloca !cir.ptr<!rec_DelegatingWithZeroing>, !cir.ptr<!cir.ptr<!rec_DelegatingWithZeroing>>, ["this", init] {alignment = 8 : i64}
// CHECK-NEXT:    %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["", init] {alignment = 4 : i64}
// CHECK-NEXT:    cir.store{{.*}} %arg0, %0 : !cir.ptr<!rec_DelegatingWithZeroing>, !cir.ptr<!cir.ptr<!rec_DelegatingWithZeroing>>
// CHECK-NEXT:    cir.store{{.*}} %arg1, %1 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:    %2 = cir.load %0 : !cir.ptr<!cir.ptr<!rec_DelegatingWithZeroing>>, !cir.ptr<!rec_DelegatingWithZeroing>
// CHECK-NEXT:    %3 = cir.const #cir.zero : !rec_DelegatingWithZeroing
// CHECK-NEXT:    cir.store{{.*}} %3, %2 : !rec_DelegatingWithZeroing, !cir.ptr<!rec_DelegatingWithZeroing>
// CHECK-NEXT:    cir.return
// CHECK-NEXT:  }

// OG uses memset instead of calling the trivial default constructor.
// CIR uses store zeroinitializer instead. Both achieve the same result.
// LLVM-LABEL: define {{.*}} @_ZN21DelegatingWithZeroingC2Ei
// LLVM-NOT:     call {{.*}} @_ZN21DelegatingWithZeroingC2Ev
// LLVM:         store %struct.DelegatingWithZeroing zeroinitializer
// LLVM:         ret void

// OGCG-LABEL: define {{.*}} @_ZN21DelegatingWithZeroingC2Ei
// OGCG-NOT:     call {{.*}} @_ZN21DelegatingWithZeroingC2Ev
// OGCG:         call void @llvm.memset
// OGCG:         ret void

void canThrow();
struct HasNonTrivialDestructor {
  HasNonTrivialDestructor();
  HasNonTrivialDestructor(int);
  ~HasNonTrivialDestructor();
};

// Check that we call the destructor whenever a cleanup is needed.
// FIXME: enable and check this when exceptions are fully supported.
#if 0
HasNonTrivialDestructor::HasNonTrivialDestructor(int)
    : HasNonTrivialDestructor() {
  canThrow();
}
#endif

// From clang/test/CodeGenCXX/cxx0x-delegating-ctors.cpp, check that virtual
// inheritance and delegating constructors interact correctly.
// FIXME: enable and check this when virtual inheritance is fully supported.
#if 0
namespace PR14588 {
void other();

class Base {
public:
  Base() { squawk(); }
  virtual ~Base() {}

  virtual void squawk() { other(); }
};

class Foo : public virtual Base {
public:
  Foo();
  Foo(const void *inVoid);
  virtual ~Foo() {}

  virtual void squawk() { other(); }
};

Foo::Foo() : Foo(nullptr) { other(); }
Foo::Foo(const void *inVoid) { squawk(); }
} // namespace PR14588
#endif
