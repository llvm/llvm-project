// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fstrict-vtable-pointers -O1 \
// RUN:     -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fstrict-vtable-pointers -O1 \
// RUN:     -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fstrict-vtable-pointers -O1 \
// RUN:     -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

void some_function();

struct Base {
  virtual ~Base();
};

struct Derived : Base {
  virtual ~Derived();
};

Derived::~Derived() { some_function(); }

// CIR-LABEL: cir.func {{.*}} @_ZN7DerivedD2Ev(
// CIR-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca "this" {{.*}} init
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR-NEXT:   %[[LAUNDERED:.*]] = cir.launder %[[THIS]]
// CIR-NEXT:   cir.cleanup.scope {
// CIR-NEXT:     %[[DERIVED_VPTR:.*]] = cir.vtable.address_point(@_ZTV7Derived, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR-NEXT:     %[[DERIVED_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[LAUNDERED]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!cir.vptr>
// CIR-NEXT:     cir.store{{.*}} %[[DERIVED_VPTR]], %[[DERIVED_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR-NEXT:     cir.call @_Z13some_functionv()
// CIR-NEXT:     cir.yield
// CIR-NEXT:   } cleanup normal {
// CIR-NEXT:     %[[BASE_ADDR:.*]] = cir.base_class_addr %[[LAUNDERED]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR-NEXT:     cir.call @_ZN4BaseD2Ev(%[[BASE_ADDR]])
// CIR-NEXT:     cir.yield
// CIR-NEXT:   }
// CIR-NEXT:   cir.return

// LLVM-LABEL: define{{.*}} void @_ZN7DerivedD2Ev(
// LLVM:        %[[LAUNDERED:.*]] = {{.*}}call ptr @llvm.launder.invariant.group.p0(ptr {{.*}})
// LLVMCIR-NEXT:store ptr getelementptr inbounds nuw (i8, ptr @_ZTV7Derived, i64 16), ptr %[[LAUNDERED]]
// OGCG-NEXT:   store ptr getelementptr inbounds nuw inrange(-16, 16) (i8, ptr @_ZTV7Derived, i64 16), ptr %[[LAUNDERED]]
// LLVM:        call void @_Z13some_functionv()
// LLVM:        call void @_ZN4BaseD2Ev(ptr {{.*}}%[[LAUNDERED]])
// LLVM:        ret void
