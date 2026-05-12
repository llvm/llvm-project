// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// __builtin_launder doesn't actually DO anything unless fstrict-vtable-pointers
// is enabled, so test here to make sure we get that switch correctly done.
// RUN: %clang_cc1 -fstrict-vtable-pointers -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR-STRICT
// RUN: %clang_cc1 -fstrict-vtable-pointers -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM-STRICT
// RUN: %clang_cc1 -fstrict-vtable-pointers -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG-STRICT


struct Base {};

struct Derived : virtual Base {};

struct VirtMem {
  Derived d;
};

auto use_derived(Derived *d) {
  return __builtin_launder(d);
}
// CIR-LABEL: cir.func{{.*}}@_Z11use_derivedP7Derived
// CIR: %[[ARG_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>, ["d", init]
// CIR: %[[RET_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>, ["__retval"]
// CIR: %[[ARG_LOAD:.*]] = cir.load align(8) %[[ARG_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_Derived>>, !cir.ptr<!rec_Derived>
// CIR: cir.store %[[ARG_LOAD]], %[[RET_ALLOCA]] : !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>
// CIR: %[[RET:.*]] = cir.load %[[RET_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_Derived>>, !cir.ptr<!rec_Derived>
// CIR: cir.return %[[RET]] : !cir.ptr<!rec_Derived>
//
// LLVM-LABEL: define {{.*}}@_Z11use_derivedP7Derived
// LLVM: %[[ARG_ALLOCA:.*]] = alloca ptr
// LLVM: %[[RET_ALLOCA:.*]] = alloca ptr
// LLVM: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG_ALLOCA]]
// LLVM: store ptr %[[ARG_LOAD]], ptr %[[RET_ALLOCA]]
// LLVM: %[[RET:.*]] = load ptr, ptr %[[RET_ALLOCA]]
// LLVM: ret ptr %[[RET]]
//
// OGCG-LABEL: define {{.*}}@_Z11use_derivedP7Derived
// OGCG: %[[ARG_ALLOCA:.*]] = alloca ptr
// OGCG: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG_ALLOCA]]
// OGCG: ret ptr %[[ARG_LOAD]]
//
// CIR-STRICT-LABEL: cir.func{{.*}}@_Z11use_derivedP7Derived
// CIR-STRICT: %[[ARG_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>, ["d", init]
// CIR-STRICT: %[[RET_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>, ["__retval"]
// CIR-STRICT: %[[LOAD_ARG:.*]] = cir.load align(8) %[[ARG_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_Derived>>, !cir.ptr<!rec_Derived>
// CIR-STRICT: %[[LAUNDER:.*]] = cir.launder %[[LOAD_ARG]]  : !cir.ptr<!rec_Derived>
// CIR-STRICT: cir.store %[[LAUNDER]], %[[RET_ALLOCA]] : !cir.ptr<!rec_Derived>, !cir.ptr<!cir.ptr<!rec_Derived>>
// CIR-STRICT: %[[RET:.*]] = cir.load %[[RET_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_Derived>>, !cir.ptr<!rec_Derived>
// CIR-STRICT: cir.return %[[RET]] : !cir.ptr<!rec_Derived>
//
// LLVM-STRICT-LABEL: define {{.*}}@_Z11use_derivedP7Derived
// LLVM-STRICT: %[[ARG_ALLOCA:.*]] = alloca ptr
// LLVM-STRICT: %[[RET_ALLOCA:.*]] = alloca ptr
// LLVM-STRICT: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG_ALLOCA]]
// LLVM-STRICT: %[[LAUNDER:.*]] = call ptr @llvm.launder.invariant.group.p0(ptr %[[ARG_LOAD]])
// LLVM-STRICT: store ptr %[[LAUNDER]], ptr %[[RET_ALLOCA]]
// LLVM-STRICT: %[[RET:.*]] = load ptr, ptr %[[RET_ALLOCA]]
// LLVM-STRICT: ret ptr %[[RET]]
//
//
// OGCG-STRICT-LABEL: define {{.*}}@_Z11use_derivedP7Derived
// OGCG-STRICT: %[[ARG_ALLOCA:.*]] = alloca ptr
// OGCG-STRICT: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG_ALLOCA]]
// OGCG-STRICT: %[[LAUNDER:.*]] = call ptr @llvm.launder.invariant.group.p0(ptr %[[ARG_LOAD]])
// OGCG-STRICT: ret ptr %[[LAUNDER]]

auto use_vm(VirtMem *vm) {
  return __builtin_launder(vm);
}
// CIR-LABEL: cir.func {{.*}}@_Z6use_vmP7VirtMem
// CIR: %[[ARG_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_VirtMem>, !cir.ptr<!cir.ptr<!rec_VirtMem>>, ["vm", init]
// CIR: %[[RET_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_VirtMem>, !cir.ptr<!cir.ptr<!rec_VirtMem>>, ["__retval"]
// CIR: cir.store %[[ARG_LOAD]], %[[RET_ALLOCA]] : !cir.ptr<!rec_VirtMem>, !cir.ptr<!cir.ptr<!rec_VirtMem>>
// CIR: %[[RET:.*]] = cir.load %[[RET_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_VirtMem>>, !cir.ptr<!rec_VirtMem>
// CIR: cir.return %[[RET]] : !cir.ptr<!rec_VirtMem>
//
// LLVM-LABEL: define {{.*}}@_Z6use_vmP7VirtMem
// LLVM: %[[ARG_ALLOCA:.*]] = alloca ptr
// LLVM: %[[RET_ALLOCA:.*]] = alloca ptr
// LLVM: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG_ALLOCA]]
// LLVM: store ptr %[[ARG_LOAD]], ptr %[[RET_ALLOCA]]
// LLVM: %[[RET:.*]] = load ptr, ptr %[[RET_ALLOCA]]
// LLVM: ret ptr %[[RET]]
//
// OGCG-LABEL: define {{.*}}@_Z6use_vmP7VirtMem
// OGCG: %[[ARG_ALLOCA:.*]] = alloca ptr
// OGCG: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG_ALLOCA]]
// OGCG: ret ptr %[[ARG_LOAD]]
//
// CIR-STRICT-LABEL: cir.func {{.*}}@_Z6use_vmP7VirtMem
// CIR-STRICT: %[[ARG_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_VirtMem>, !cir.ptr<!cir.ptr<!rec_VirtMem>>, ["vm", init]
// CIR-STRICT: %[[RET_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_VirtMem>, !cir.ptr<!cir.ptr<!rec_VirtMem>>, ["__retval"]
// CIR-STRICT: %[[LOAD_ARG:.*]] = cir.load align(8) %[[ARG_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_VirtMem>>, !cir.ptr<!rec_VirtMem>
// CIR-STRICT: %[[LAUNDER:.*]] = cir.launder %[[LOAD_ARG]] : !cir.ptr<!rec_VirtMem>
// CIR-STRICT: cir.store %[[LAUNDER]], %[[RET_ALLOCA]] : !cir.ptr<!rec_VirtMem>, !cir.ptr<!cir.ptr<!rec_VirtMem>>
// CIR-STRICT: %[[RET:.*]] = cir.load %[[RET_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_VirtMem>>, !cir.ptr<!rec_VirtMem>
// CIR-STRICT: cir.return %[[RET]] : !cir.ptr<!rec_VirtMem>
//
// LLVM-STRICT-LABEL: define {{.*}}@_Z6use_vmP7VirtMem
// LLVM-STRICT: %[[ARG_ALLOCA:.*]] = alloca ptr
// LLVM-STRICT: %[[RET_ALLOCA:.*]] = alloca ptr
// LLVM-STRICT: %[[ARG_LOAD:.*]] = load ptr, ptr %[[ARG_ALLOCA]]
// LLVM-STRICT: %[[LAUNDER:.*]] = call ptr @llvm.launder.invariant.group.p0(ptr %[[ARG_LOAD]])
// LLVM-STRICT: store ptr %[[LAUNDER]], ptr %[[RET_ALLOCA]]
// LLVM-STRICT: %[[RET:.*]] = load ptr, ptr %[[RET_ALLOCA]]
// LLVM-STRICT: ret ptr %[[RET]]
//
// OGCG-STRICT-LABEL: define {{.*}}@_Z6use_vmP7VirtMem
// OGCG-STRICT: %[[ARG_ALLOCA:.*]] = alloca ptr
// OGCG-STRICT: %[[LAUNDER:.*]] = call ptr @llvm.launder.invariant.group.p0(ptr %[[ARG_LOAD]])
// OGCG-STRICT: ret ptr %[[LAUNDER]]
