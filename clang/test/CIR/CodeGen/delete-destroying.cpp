// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Minimal in-source declarations of the standard-library bits needed for a
// destroying operator delete so this test does not depend on a system header.
namespace std {
struct destroying_delete_t {
  explicit destroying_delete_t() = default;
};
inline constexpr destroying_delete_t destroying_delete{};
} // namespace std

// A destroying operator delete: its second parameter is
// std::destroying_delete_t. Such an operator delete takes full responsibility
// for both destruction and deallocation of the object, so the 'delete'
// expression must not emit a separate destructor call or cleanup.
struct S {
  void operator delete(S *, std::destroying_delete_t);
  ~S();
};

void test_destroying_delete(S *s) {
  delete s;
}

// S::operator delete(S *, std::destroying_delete_t)
// CIR: cir.func private @_ZN1SdlEPS_St19destroying_delete_t(!cir.ptr<!rec_S> {llvm.noundef}, !rec_std3A3Adestroying_delete_t)
// LLVM: declare void @_ZN1SdlEPS_St19destroying_delete_t(ptr noundef, %"struct.std::destroying_delete_t")

// The destroying operator delete takes over the entire delete operation:
// no destructor call and no delete-cleanup are emitted in the caller; the
// operator delete is invoked inside the standard null-check if-region and is
// responsible for destroying the object itself.

// CIR: cir.func {{.*}} @_Z22test_destroying_deleteP1S(%[[ARG:.*]]: !cir.ptr<!rec_S> {{.*}})
// CIR:   %[[S_ADDR:.*]] = cir.alloca !cir.ptr<!rec_S>, {{.*}} ["s", init]
// CIR:   %[[TAG_ADDR:.*]] = cir.alloca !rec_std3A3Adestroying_delete_t, {{.*}} ["destroying.delete.tag"]
// CIR:   cir.store %[[ARG]], %[[S_ADDR]]
// CIR:   %[[S:.*]] = cir.load{{.*}} %[[S_ADDR]]
// CIR:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_S>
// CIR:   %[[NOT_NULL:.*]] = cir.cmp ne %[[S]], %[[NULL]] : !cir.ptr<!rec_S>
// CIR:   cir.if %[[NOT_NULL]] {
// CIR:     %[[TAG:.*]] = cir.load{{.*}} %[[TAG_ADDR]]
// CIR:     cir.call @_ZN1SdlEPS_St19destroying_delete_t(%[[S]], %[[TAG]]){{.*}} : (!cir.ptr<!rec_S> {{.*}}, !rec_std3A3Adestroying_delete_t) -> ()
// CIR-NOT: cir.call @_ZN1SD{{[12]}}Ev
// CIR:   }
// CIR:   cir.return

// LLVM: define {{.*}} void @_Z22test_destroying_deleteP1S(ptr {{.*}} %[[ARG:.*]])
// LLVM:   %[[S_ADDR:.*]] = alloca ptr
// LLVM:   %[[TAG_ADDR:.*]] = alloca %"struct.std::destroying_delete_t"
// LLVM:   store ptr %[[ARG]], ptr %[[S_ADDR]]
// LLVM:   %[[S:.*]] = load ptr, ptr %[[S_ADDR]]
// LLVM:   %[[NOT_NULL:.*]] = icmp ne ptr %[[S]], null
// LLVM:   br i1 %[[NOT_NULL]], label %[[NOTNULL:.*]], label %[[END:.*]]
// LLVM: [[NOTNULL]]:
// LLVM:   %[[TAG:.*]] = load %"struct.std::destroying_delete_t", ptr %[[TAG_ADDR]]
// LLVM:   call void @_ZN1SdlEPS_St19destroying_delete_t(ptr noundef %[[S]], %"struct.std::destroying_delete_t" %[[TAG]])
// LLVM-NOT: call void @_ZN1SD{{[12]}}Ev
// LLVM: [[END]]:
// LLVM:   ret void

// Classic codegen elides empty-class parameters at the ABI level, so the
// destroying_delete_t tag disappears from both the declaration and the call.
// Either way, no destructor call is emitted from the caller.
// OGCG: define {{.*}} void @_Z22test_destroying_deleteP1S(ptr {{.*}} %[[ARG:.*]])
// OGCG:   %[[S_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG]], ptr %[[S_ADDR]]
// OGCG:   %[[S:.*]] = load ptr, ptr %[[S_ADDR]]
// OGCG:   %[[ISNULL:.*]] = icmp eq ptr %[[S]], null
// OGCG:   br i1 %[[ISNULL]], label %[[END:.*]], label %[[NOTNULL:.*]]
// OGCG: [[NOTNULL]]:
// OGCG:   call void @_ZN1SdlEPS_St19destroying_delete_t(ptr noundef %[[S]])
// OGCG-NOT: call void @_ZN1SD{{[12]}}Ev
// OGCG: [[END]]:
// OGCG:   ret void
// OGCG: declare void @_ZN1SdlEPS_St19destroying_delete_t(ptr noundef)

// A class with a virtual destructor and a destroying operator delete.
// Per the Itanium C++ ABI, the call is dispatched through the vtable's
// deleting-destructor slot (entry index 1); that function is responsible
// for running the destructor chain and then invoking the class-level
// destroying operator delete. The caller therefore does not call the
// destructor or the operator delete directly.
struct V {
  virtual ~V();
  void operator delete(V *, std::destroying_delete_t);
};

void test_virtual_destroying_delete(V *v) {
  delete v;
}

// CIR: cir.func {{.*}} @_Z30test_virtual_destroying_deleteP1V(%[[ARG:.*]]: !cir.ptr<!rec_V> {{.*}})
// CIR:   %[[V_ADDR:.*]] = cir.alloca !cir.ptr<!rec_V>, {{.*}} ["v", init]
// CIR:   cir.store %[[ARG]], %[[V_ADDR]]
// CIR:   %[[V:.*]] = cir.load{{.*}} %[[V_ADDR]]
// CIR:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!rec_V>
// CIR:   %[[NOT_NULL:.*]] = cir.cmp ne %[[V]], %[[NULL]] : !cir.ptr<!rec_V>
// CIR:   cir.if %[[NOT_NULL]] {
// CIR:     %[[VPTR_ADDR:.*]] = cir.vtable.get_vptr %[[V]] : !cir.ptr<!rec_V> -> !cir.ptr<!cir.vptr>
// CIR:     %[[VPTR:.*]] = cir.load{{.*}} %[[VPTR_ADDR]] : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR:     %[[SLOT_ADDR:.*]] = cir.vtable.get_virtual_fn_addr %[[VPTR]][1]
// CIR:     %[[SLOT:.*]] = cir.load{{.*}} %[[SLOT_ADDR]]
// CIR:     cir.call %[[SLOT]](%[[V]]){{.*}} : (!cir.ptr<!cir.func<(!cir.ptr<!rec_V>)>>, !cir.ptr<!rec_V> {{.*}}) -> ()
// CIR-NOT: cir.call @_ZN1VdlEPS_St19destroying_delete_t
// CIR-NOT: cir.call @_ZN1VD{{[12]}}Ev
// CIR:   }
// CIR:   cir.return

// LLVM: define {{.*}} void @_Z30test_virtual_destroying_deleteP1V(ptr {{.*}} %[[ARG:.*]])
// LLVM:   %[[V_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG]], ptr %[[V_ADDR]]
// LLVM:   %[[V:.*]] = load ptr, ptr %[[V_ADDR]]
// LLVM:   %[[NOT_NULL:.*]] = icmp ne ptr %[[V]], null
// LLVM:   br i1 %[[NOT_NULL]], label %[[NOTNULL:.*]], label %[[END:.*]]
// LLVM: [[NOTNULL]]:
// LLVM:   %[[VTABLE:.*]] = load ptr, ptr %[[V]]
// LLVM:   %[[VFN_PTR:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE]], i32 1
// LLVM:   %[[VFN:.*]] = load ptr, ptr %[[VFN_PTR]]
// LLVM:   call void %[[VFN]](ptr {{.*}} %[[V]])
// LLVM-NOT: call void @_ZN1VdlEPS_St19destroying_delete_t
// LLVM-NOT: call void @_ZN1VD{{[12]}}Ev
// LLVM: [[END]]:
// LLVM:   ret void

// OGCG: define {{.*}} void @_Z30test_virtual_destroying_deleteP1V(ptr {{.*}} %[[ARG:.*]])
// OGCG:   %[[V_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG]], ptr %[[V_ADDR]]
// OGCG:   %[[V:.*]] = load ptr, ptr %[[V_ADDR]]
// OGCG:   %[[ISNULL:.*]] = icmp eq ptr %[[V]], null
// OGCG:   br i1 %[[ISNULL]], label %[[END:.*]], label %[[NOTNULL:.*]]
// OGCG: [[NOTNULL]]:
// OGCG:   %[[VTABLE:.*]] = load ptr, ptr %[[V]]
// OGCG:   %[[VFN_PTR:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE]], i64 1
// OGCG:   %[[VFN:.*]] = load ptr, ptr %[[VFN_PTR]]
// OGCG:   call void %[[VFN]](ptr {{.*}} %[[V]])
// OGCG-NOT: call void @_ZN1VdlEPS_St19destroying_delete_t
// OGCG-NOT: call void @_ZN1VD{{[12]}}Ev
// OGCG: [[END]]:
// OGCG:   ret void
