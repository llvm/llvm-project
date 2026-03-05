// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t.cir
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.cir
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.cir %s

class A {
  virtual ~A();
  A B(A);
};
A A::B(A) {
  // CIR-LABEL:   cir.func {{.*}} @_ZN1A1BES_(
  // CIR-SAME:      %[[THIS_ARG:.*]]: !cir.ptr<!rec_A>
  // CIR-NEXT:           %[[THIS_VAR:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
  // CIR:                cir.store %[[THIS_ARG]], %[[THIS_VAR]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
  // CIR:                %[[THIS:.*]] = cir.load %[[THIS_VAR]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
  // CIR-NEXT:           %[[VPTR_PTR:.*]] = cir.vtable.get_vptr %[[THIS]] : !cir.ptr<!rec_A> -> !cir.ptr<!cir.vptr>
  // CIR-NEXT:           %[[VPTR:.*]] = cir.load align(8) %[[VPTR_PTR]] : !cir.ptr<!cir.vptr>, !cir.vptr
  // CIR-NEXT:           %[[DTOR_PTR:.*]] = cir.vtable.get_virtual_fn_addr %[[VPTR]][0] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>>
  // CIR-NEXT:           %[[DTOR:.*]] = cir.load align(8) %[[DTOR_PTR]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>
  // CIR-NEXT:           cir.call %[[DTOR]](%[[THIS]]) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_A>)>>, !cir.ptr<!rec_A>) -> ()
  // CIR-NEXT:           cir.trap
  // CIR-NEXT:         }


  // LLVM-LABEL:   define dso_local %class.A @_ZN1A1BES_(
  // LLVM-SAME:      ptr %[[THIS_ARG:[0-9]+]],
  // LLVM-NEXT:          %[[THIS_VAR:.*]] = alloca ptr, i64 1, align 8
  // LLVM:               store ptr %[[THIS_ARG]], ptr %[[THIS_VAR]], align 8
  // LLVM:               %[[THIS:.*]] = load ptr, ptr %[[THIS_VAR]], align 8
  // LLVM-NEXT:          %[[VTABLE_PTR:.*]] = load ptr, ptr %[[THIS]], align 8
  // LLVM-NEXT:          %[[VIRT_DTOR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE_PTR]], i32 0
  // LLVM-NEXT:          %[[DTOR:.*]] = load ptr, ptr %[[VIRT_DTOR_ADDR]], align 8
  // LLVM-NEXT:          call void %[[DTOR]](ptr %[[THIS]])
  // LLVM-NEXT:          call void @llvm.trap()
  // LLVM-NEXT:          unreachable
  // LLVM-NEXT:        }

  
  // OGCG-LABEL:   define dso_local void @_ZN1A1BES_(
  // OGCG-SAME:      ptr {{.*}}%[[THIS_ARG:.*]],
  // OGCG:               %[[VAL_0:.*]] = alloca ptr, align 8
  // OGCG-NEXT:          %[[THIS_VAR:.*]] = alloca ptr, align 8
  // OGCG:               store ptr %[[THIS_ARG]], ptr %[[THIS_VAR]], align 8
  // OGCG:               %[[THIS:.*]] = load ptr, ptr %[[THIS_VAR]], align 8
  // OGCG-NEXT:          %[[VTABLE:.*]] = load ptr, ptr %[[THIS]], align 8
  // OGCG-NEXT:          %[[VIRT_DTOR_ADDR:.*]] = getelementptr inbounds ptr, ptr %[[VTABLE]], i64 0
  // OGCG-NEXT:          %[[DTOR:.*]] = load ptr, ptr %[[VIRT_DTOR_ADDR]], align 8
  // OGCG-NEXT:          call void %[[DTOR]](ptr noundef nonnull align 8 dereferenceable(8) %[[THIS]]) #2
  // OGCG-NEXT:          call void @llvm.trap()
  // OGCG-NEXT:          unreachable
  // OGCG-NEXT:        }

  this->~A();
}
