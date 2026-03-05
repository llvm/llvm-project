// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -relaxed-aliasing
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O0
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s

// NO-TBAA-NOT: !tbaa

// CIR: #tbaa[[VPTR:.*]] = #cir.tbaa_vptr<type = !cir.vptr>

struct Member {
  ~Member();
};

struct A {
  virtual ~A();
};

struct B : A {
  Member m;
  virtual ~B();
};
B::~B() { }

// CIR-LABEL: _ZN1BD2Ev
// CIR: cir.store{{.*}} %{{.*}}, %{{.*}} : !cir.vptr, !cir.ptr<!cir.vptr> tbaa(#tbaa[[VPTR]])

// LLVM-LABEL: _ZN1BD2Ev
// LLVM: store ptr getelementptr inbounds nuw (i8, ptr @_ZTV1B, i64 16), ptr %{{.*}}, align 8, !tbaa ![[TBAA_VPTR:.*]]
// LLVM: ![[TBAA_VPTR]] = !{![[TBAA_VPTR_PARENT:.*]], ![[TBAA_VPTR_PARENT]], i64 0}
// LLVM: ![[TBAA_VPTR_PARENT]] = !{!"vtable pointer", !
