// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

typedef __typeof(sizeof(int)) size_t;

struct SizedDelete {
  void operator delete(void*, size_t);
  int member;
};
void test_sized_delete(SizedDelete *x) {
  delete x;
}

// SizedDelete::operator delete(void*, unsigned long)
// CIR:  cir.func private @_ZN11SizedDeletedlEPvm(!cir.ptr<!void>, !u64i)
// LLVM: declare void @_ZN11SizedDeletedlEPvm(ptr, i64)

// CIR: cir.func dso_local @_Z17test_sized_deleteP11SizedDelete
// CIR:   %[[X:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[X_CAST:.*]] = cir.cast bitcast %[[X]] : !cir.ptr<!rec_SizedDelete> -> !cir.ptr<!void>
// CIR:   %[[OBJ_SIZE:.*]] = cir.const #cir.int<4> : !u64i
// CIR:   cir.call @_ZN11SizedDeletedlEPvm(%[[X_CAST]], %[[OBJ_SIZE]]) nothrow : (!cir.ptr<!void>, !u64i) -> ()

// LLVM: define dso_local void @_Z17test_sized_deleteP11SizedDelete
// LLVM:   %[[X:.*]] = load ptr, ptr %{{.*}}
// LLVM:   call void @_ZN11SizedDeletedlEPvm(ptr %[[X]], i64 4)

// OGCG: define dso_local void @_Z17test_sized_deleteP11SizedDelete
// OGCG:   %[[X:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[ISNULL:.*]] = icmp eq ptr %[[X]], null
// OGCG:   br i1 %[[ISNULL]], label %{{.*}}, label %[[DELETE_NOTNULL:.*]]
// OGCG: [[DELETE_NOTNULL]]:
// OGCG:   call void @_ZN11SizedDeletedlEPvm(ptr noundef %[[X]], i64 noundef 4)

// This function is declared below the call in OGCG.
// OGCG: declare void @_ZN11SizedDeletedlEPvm(ptr noundef, i64 noundef)

struct Contents {
  ~Contents() {}
};
struct Container {
  Contents *contents;
  ~Container();
};
Container::~Container() { delete contents; }

// Contents::~Contents()
// CIR: cir.func comdat linkonce_odr @_ZN8ContentsD2Ev
// LLVM: define linkonce_odr void @_ZN8ContentsD2Ev

// operator delete(void*, unsigned long)
// CIR: cir.func private @_ZdlPvm(!cir.ptr<!void>, !u64i)
// LLVM: declare void @_ZdlPvm(ptr, i64)

// Container::~Container()
// CIR: cir.func dso_local @_ZN9ContainerD2Ev
// CIR:   %[[THIS:.*]] = cir.load %{{.*}}
// CIR:   %[[CONTENTS_PTR_ADDR:.*]] = cir.get_member %[[THIS]][0] {name = "contents"} : !cir.ptr<!rec_Container> -> !cir.ptr<!cir.ptr<!rec_Contents>>
// CIR:   %[[CONTENTS_PTR:.*]] = cir.load{{.*}} %[[CONTENTS_PTR_ADDR]]
// CIR:   cir.call @_ZN8ContentsD2Ev(%[[CONTENTS_PTR]]) nothrow : (!cir.ptr<!rec_Contents>) -> ()
// CIR:   %[[CONTENTS_CAST:.*]] = cir.cast bitcast %[[CONTENTS_PTR]] : !cir.ptr<!rec_Contents> -> !cir.ptr<!void>
// CIR:   %[[OBJ_SIZE:.*]] = cir.const #cir.int<1> : !u64i
// CIR:   cir.call @_ZdlPvm(%[[CONTENTS_CAST]], %[[OBJ_SIZE]]) nothrow : (!cir.ptr<!void>, !u64i) -> ()

// LLVM: define dso_local void @_ZN9ContainerD2Ev
// LLVM:   %[[THIS:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[CONTENTS_PTR_ADDR:.*]] = getelementptr %struct.Container, ptr %[[THIS]], i32 0, i32 0
// LLVM:   %[[CONTENTS_PTR:.*]] = load ptr, ptr %[[CONTENTS_PTR_ADDR]]
// LLVM:   call void @_ZN8ContentsD2Ev(ptr %[[CONTENTS_PTR]])
// LLVM:   call void @_ZdlPvm(ptr %[[CONTENTS_PTR]], i64 1)

// OGCG: define dso_local void @_ZN9ContainerD2Ev
// OGCG:   %[[THIS:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[CONTENTS:.*]] = getelementptr inbounds nuw %struct.Container, ptr %[[THIS]], i32 0, i32 0
// OGCG:   %[[CONTENTS_PTR:.*]] = load ptr, ptr %[[CONTENTS]]
// OGCG:   %[[ISNULL:.*]] = icmp eq ptr %[[CONTENTS_PTR]], null
// OGCG:   br i1 %[[ISNULL]], label %{{.*}}, label %[[DELETE_NOTNULL:.*]]
// OGCG: [[DELETE_NOTNULL]]:
// OGCG:   call void @_ZN8ContentsD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[CONTENTS_PTR]])
// OGCG:   call void @_ZdlPvm(ptr noundef %[[CONTENTS_PTR]], i64 noundef 1)

// These functions are declared/defined below the calls in OGCG.
// OGCG: define linkonce_odr void @_ZN8ContentsD2Ev
// OGCG: declare void @_ZdlPvm(ptr noundef, i64 noundef)

struct StructWithVirtualDestructor {
  virtual ~StructWithVirtualDestructor();
};

void destroy(StructWithVirtualDestructor *x) {
  delete x;
}

// CIR: cir.func {{.*}} @_Z7destroyP27StructWithVirtualDestructor(%[[X_ARG:.*]]: !cir.ptr<!rec_StructWithVirtualDestructor> {{.*}})
// CIR:   %[[X_ADDR:.*]] = cir.alloca !cir.ptr<!rec_StructWithVirtualDestructor>
// CIR:   cir.store %[[X_ARG]], %[[X_ADDR]]
// CIR:   %[[X:.*]] = cir.load{{.*}} %[[X_ADDR]]
// CIR:   %[[VTABLE_PTR:.*]] = cir.vtable.get_vptr %[[X]] : !cir.ptr<!rec_StructWithVirtualDestructor> -> !cir.ptr<!cir.vptr>
// CIR:   %[[VTABLE:.*]] = cir.load{{.*}} %[[VTABLE_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:   %[[DTOR_FN_ADDR_PTR:.*]] = cir.vtable.get_virtual_fn_addr %[[VTABLE]][1]
// CIR:   %[[DTOR_FN_ADDR:.*]] = cir.load{{.*}} %[[DTOR_FN_ADDR_PTR]]
// CIR:   cir.call %[[DTOR_FN_ADDR]](%[[X]])

// LLVM: define {{.*}} void @_Z7destroyP27StructWithVirtualDestructor(ptr %[[X_ARG:.*]])
// LLVM:   %[[X_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[X_ARG]], ptr %[[X_ADDR]]
// LLVM:   %[[X:.*]] = load ptr, ptr %[[X_ADDR]]
// LLVM:   %[[VTABLE:.*]] = load ptr, ptr %[[X]]
// LLVM:   %[[DTOR_FN_ADDR_PTR:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE]], i32 1
// LLVM:   %[[DTOR_FN_ADDR:.*]] = load ptr, ptr %[[DTOR_FN_ADDR_PTR]]
// LLVM:   call void %[[DTOR_FN_ADDR]](ptr %[[X]])

// OGCG: define {{.*}} void @_Z7destroyP27StructWithVirtualDestructor(ptr {{.*}} %[[X_ARG:.*]])
// OGCG:   %[[X_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[X_ARG]], ptr %[[X_ADDR]]
// OGCG:   %[[X:.*]] = load ptr, ptr %[[X_ADDR]]
// OGCG:   %[[ISNULL:.*]] = icmp eq ptr %[[X]], null
// OGCG:   br i1 %[[ISNULL]], label %{{.*}}, label %[[DELETE_NOTNULL:.*]]
// OGCG: [[DELETE_NOTNULL]]:
// OGCG:   %[[VTABLE:.*]] = load ptr, ptr %[[X]]
// OGCG:   %[[DTOR_FN_ADDR_PTR:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE]], i64 1
// OGCG:   %[[DTOR_FN_ADDR:.*]] = load ptr, ptr %[[DTOR_FN_ADDR_PTR]]
// OGCG:   call void %[[DTOR_FN_ADDR]](ptr {{.*}} %[[X]])
