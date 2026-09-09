// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

void some_function();

struct Base {
  virtual ~Base();
};

struct Derived : Base {
  virtual ~Derived();
};

Base::~Base() { some_function(); }

// CIR-LABEL: cir.func {{.*}} @_ZN4BaseD2Ev(
// CIR-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca "this" {{.*}} init
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR-NEXT:   %[[BASE_VPTR:.*]] = cir.vtable.address_point(@_ZTV4Base, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR-NEXT:   %[[BASE_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_Base> -> !cir.ptr<!cir.vptr>
// CIR-NEXT:   cir.store{{.*}} %[[BASE_VPTR]], %[[BASE_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR-NEXT:   cir.call @_Z13some_functionv()
// CIR-NEXT:   cir.return

// LLVM-LABEL: define{{.*}} void @_ZN4BaseD2Ev(
// LLVM:        %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:        %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVMCIR-NEXT:store ptr getelementptr inbounds nuw (i8, ptr @_ZTV4Base, i64 16), ptr %[[THIS]]
// OGCG-NEXT:   store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr] }, ptr @_ZTV4Base, i32 0, i32 0, i32 2), ptr %[[THIS]]
// LLVM-NEXT:   call void @_Z13some_functionv()
// LLVM-NEXT:   ret void


Derived::~Derived() { some_function(); }

// CIR-LABEL: cir.func {{.*}} @_ZN7DerivedD2Ev(
// CIR-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca "this" {{.*}} init
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR-NEXT:   cir.cleanup.scope {
// CIR-NEXT:     %[[DERIVED_VPTR:.*]] = cir.vtable.address_point(@_ZTV7Derived, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR-NEXT:     %[[DERIVED_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_Derived> -> !cir.ptr<!cir.vptr>
// CIR-NEXT:     cir.store{{.*}} %[[DERIVED_VPTR]], %[[DERIVED_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR-NEXT:     cir.call @_Z13some_functionv()
// CIR-NEXT:     cir.yield
// CIR-NEXT:   } cleanup normal {
// CIR-NEXT:     %[[BASE_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_Derived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR-NEXT:     cir.call @_ZN4BaseD2Ev(%[[BASE_ADDR]])
// CIR-NEXT:     cir.yield
// CIR-NEXT:   }
// CIR-NEXT:   cir.return

// Cleanup scopes insert a bunch of empty blocks, so we can't use LLVM-NEXT as
// aggressively as I'd like.
// LLVM-LABEL: define{{.*}} void @_ZN7DerivedD2Ev(
// LLVM:        %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:        %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVMCIR:     store ptr getelementptr inbounds nuw (i8, ptr @_ZTV7Derived, i64 16), ptr %[[THIS]]
// OGCG:        store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr] }, ptr @_ZTV7Derived, i32 0, i32 0, i32 2), ptr %[[THIS]]
// LLVM-NEXT:   call void @_Z13some_functionv()
// LLVM:        call void @_ZN4BaseD2Ev(ptr {{.*}}%[[THIS]])
// LLVM:        ret void


// A destructor of an effectively-final class never needs to reinitialize its
// vtable pointer, since it's already known to point at the class's own
// vtable.
struct FinalDerived final : Base {
  virtual ~FinalDerived();
};

FinalDerived::~FinalDerived() { some_function(); }

// CIR-LABEL: cir.func {{.*}} @_ZN12FinalDerivedD2Ev(
// CIR-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca "this" {{.*}} init
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR-NEXT:   cir.cleanup.scope {
// CIR-NEXT:     cir.call @_Z13some_functionv()
// CIR-NEXT:     cir.yield
// CIR-NEXT:   } cleanup normal {
// CIR-NEXT:     %[[BASE_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_FinalDerived> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR-NEXT:     cir.call @_ZN4BaseD2Ev(%[[BASE_ADDR]])
// CIR-NEXT:     cir.yield
// CIR-NEXT:   }
// CIR-NEXT:   cir.return

// LLVM-LABEL: define{{.*}} void @_ZN12FinalDerivedD2Ev(
// LLVM:        %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:        %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM-NOT:    store ptr {{.*}}@_ZTV12FinalDerived
// LLVM:        call void @_Z13some_functionv()
// LLVM:        call void @_ZN4BaseD2Ev(ptr {{.*}}%[[THIS]])
// LLVM:        ret void


// A destructor with a trivial body (and no non-trivial field destructors)
// also never needs to reinitialize the vtable pointer.
struct TrivialDtor : Base {
  virtual ~TrivialDtor();
};

TrivialDtor::~TrivialDtor() {}

// CIR-LABEL: cir.func {{.*}} @_ZN11TrivialDtorD2Ev(
// CIR-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca "this" {{.*}} init
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR-NEXT:   cir.cleanup.scope {
// CIR-NEXT:     cir.yield
// CIR-NEXT:   } cleanup normal {
// CIR-NEXT:     %[[BASE_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_TrivialDtor> nonnull [0] -> !cir.ptr<!rec_Base>
// CIR-NEXT:     cir.call @_ZN4BaseD2Ev(%[[BASE_ADDR]])
// CIR-NEXT:     cir.yield
// CIR-NEXT:   }
// CIR-NEXT:   cir.return

// LLVM-LABEL: define{{.*}} void @_ZN11TrivialDtorD2Ev(
// LLVM:        %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:        %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM-NOT:    store ptr {{.*}}@_ZTV11TrivialDtor
// LLVM:        call void @_ZN4BaseD2Ev(ptr {{.*}}%[[THIS]])
// LLVM:        ret void


// A class with more than one non-virtual polymorphic base has more than one
// vtable pointer of its own to reinitialize.
struct Mother {
  virtual ~Mother();
};
struct Father {
  virtual ~Father();
};
struct MultiBase : Mother, Father {
  virtual ~MultiBase();
};

MultiBase::~MultiBase() { some_function(); }

// CIR-LABEL: cir.func {{.*}} @_ZN9MultiBaseD2Ev(
// CIR-NEXT:   %[[THIS_ADDR:.*]] = cir.alloca "this" {{.*}} init
// CIR:        %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR-NEXT:   cir.cleanup.scope {
// CIR-NEXT:     cir.cleanup.scope {
// CIR-NEXT:       %[[MOTHER_VPTR:.*]] = cir.vtable.address_point(@_ZTV9MultiBase, address_point = <index = 0, offset = 2>) : !cir.vptr
// CIR-NEXT:       %[[MOTHER_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_MultiBase> -> !cir.ptr<!cir.vptr>
// CIR-NEXT:       cir.store{{.*}} %[[MOTHER_VPTR]], %[[MOTHER_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR-NEXT:       %[[FATHER_VPTR:.*]] = cir.vtable.address_point(@_ZTV9MultiBase, address_point = <index = 1, offset = 2>) : !cir.vptr
// CIR-NEXT:       %[[FATHER_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_MultiBase> nonnull [8] -> !cir.ptr<!rec_Father>
// CIR-NEXT:       %[[FATHER_VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[FATHER_ADDR]] : !cir.ptr<!rec_Father> -> !cir.ptr<!cir.vptr>
// CIR-NEXT:       cir.store{{.*}} %[[FATHER_VPTR]], %[[FATHER_VPTR_ADDR]] : !cir.vptr, !cir.ptr<!cir.vptr>
// CIR-NEXT:       cir.call @_Z13some_functionv()
// CIR-NEXT:       cir.yield
// CIR-NEXT:     } cleanup normal {
// CIR-NEXT:       %[[FATHER_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_MultiBase> nonnull [8] -> !cir.ptr<!rec_Father>
// CIR-NEXT:       cir.call @_ZN6FatherD2Ev(%[[FATHER_ADDR]])
// CIR-NEXT:       cir.yield
// CIR-NEXT:     }
// CIR-NEXT:     cir.yield
// CIR-NEXT:   } cleanup normal {
// CIR-NEXT:     %[[MOTHER_ADDR:.*]] = cir.base_class_addr %[[THIS]] : !cir.ptr<!rec_MultiBase> nonnull [0] -> !cir.ptr<!rec_Mother>
// CIR-NEXT:     cir.call @_ZN6MotherD2Ev(%[[MOTHER_ADDR]])
// CIR-NEXT:     cir.yield
// CIR-NEXT:   }
// CIR-NEXT:   cir.return

// LLVM-LABEL: define{{.*}} void @_ZN9MultiBaseD2Ev(
// LLVM:        %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:        %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVMCIR:     store ptr getelementptr inbounds nuw (i8, ptr @_ZTV9MultiBase, i64 16), ptr %[[THIS]]
// OGCG:        store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr], [4 x ptr] }, ptr @_ZTV9MultiBase, i32 0, i32 0, i32 2), ptr %[[THIS]]
// LLVM:        %[[FATHER_ADDR:.*]] = getelementptr {{.*}}i8, ptr %[[THIS]], i{{32|64}} 8
// LLVMCIR:     store ptr getelementptr inbounds nuw (i8, ptr @_ZTV9MultiBase, i64 48), ptr %[[FATHER_ADDR]]
// OGCG:        store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr], [4 x ptr] }, ptr @_ZTV9MultiBase, i32 0, i32 1, i32 2), ptr %[[FATHER_ADDR]]
// LLVM:        call void @_Z13some_functionv()
// LLVM:        call void @_ZN6FatherD2Ev(ptr {{.*}})
// LLVM:        call void @_ZN6MotherD2Ev(ptr {{.*}}%[[THIS]])
// LLVM:        ret void
