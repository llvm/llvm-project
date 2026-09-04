// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck %s --input-file=%t-before.cir --check-prefixes=CIR
// RUN: FileCheck %s --input-file=%t.cir --check-prefixes=CIR
// RUN: %clang_cc1 %s -triple %itanium_abi_triple -Wno-unused-value -std=c++11 -fclangir -emit-llvm -o - -std=c++11 | FileCheck %s --check-prefixes=CIR-TO-LLVM
// RUN: %clang_cc1 %s -triple %itanium_abi_triple -Wno-unused-value -std=c++11 -emit-llvm -o - -std=c++11 | FileCheck %s --check-prefixes=LLVM

// FIXME: missing `inbounds` in `getelementptr` in CIR-TO-LLVM output.

namespace std {
  class type_info {};
}

struct Base {
  virtual int foo() { return 42; }
  virtual ~Base();
};

struct NonFinal : Base {};
struct Final final : Base {
    int foo() override { return 84; }
};

void func();
void ForceLoadingVTable(const std::type_info &);

// Most derived
void base_by_value(Base b) { typeid(b); }
// CIR-LABEL:  cir.func {{.*}}@_Z13base_by_value4Base
// CIR-NOT:    cir.vtable.get_vptr
// CIR:        cir.return
// LLVM-LABEL: define {{.*}}void @_Z13base_by_value4Base
// LLVM-NOT:   getelementptr
// LLVM:       ret void
// CIR-TO-LLVM-LABEL: define {{.*}}void @_Z13base_by_value4Base
// CIR-TO-LLVM-NOT:   getelementptr
// CIR-TO-LLVM:       ret void

// Most derived
void final_ref(Final &f) { typeid(f); }
// CIR-LABEL:  cir.func {{.*}}@_Z9final_refR5Final
// CIR-NOT:    cir.vtable.get_vptr
// CIR:        cir.return
// LLVM-LABEL: define {{.*}}void @_Z9final_refR5Final
// LLVM-NOT:   getelementptr
// LLVM:       ret void
// CIR-TO-LLVM-LABEL: define {{.*}}void @_Z9final_refR5Final
// CIR-TO-LLVM-NOT:   getelementptr
// CIR-TO-LLVM:       ret void

// Most derived
void final_deref(Final *f) { typeid(*f); }
// CIR-LABEL:  cir.func {{.*}}@_Z11final_derefP5Final
// CIR-NOT:    cir.vtable.get_vptr
// CIR:        cir.cmp
// CIR-NEXT:   cir.if
// CIR-NOT:    cir.vtable.get_vptr
// CIR:        cir.return
// LLVM-LABEL: define {{.*}}void @_Z11final_derefP5Final
// LLVM-NOT:   getelementptr
// LLVM:       icmp eq {{.*}}, null
// LLVM-NEXT:  br i1
// LLVM-NOT:   getelementptr
// LLVM:       ret void
// CIR-TO-LLVM-LABEL: define {{.*}}void @_Z11final_derefP5Final
// CIR-TO-LLVM-NOT:   getelementptr
// CIR-TO-LLVM:       icmp eq {{.*}}, null
// CIR-TO-LLVM-NEXT:  br i1
// CIR-TO-LLVM-NOT:   getelementptr
// CIR-TO-LLVM:       ret void

// Most derived
void should_evaluate(Final &f) { typeid(func(), f); }
// CIR-LABEL:  cir.func {{.*}}@_Z15should_evaluateR5Final
// CIR-NOT:    cir.vtable.get_vptr
// CIR:        cir.call @_Z4funcv
// CIR-NOT:    cir.vtable.get_vptr
// CIR:        cir.return
// LLVM-LABEL: define {{.*}}void @_Z15should_evaluateR5Final
// LLVM-NOT:   getelementptr
// LLVM:       call void @_Z4funcv()
// LLVM-NOT:   getelementptr
// LLVM:       ret void
// CIR-TO-LLVM-LABEL: define {{.*}}void @_Z15should_evaluateR5Final
// CIR-TO-LLVM-NOT:   getelementptr
// CIR-TO-LLVM:       call void @_Z4funcv()
// CIR-TO-LLVM-NOT:   getelementptr
// CIR-TO-LLVM:       ret void

// Not most derived
void base_ref(Base &b) { typeid(b); }
// CIR-LABEL:  cir.func {{.*}}@_Z8base_refR4Base
// CIR:        cir.vtable.get_vptr
// CIR:        cir.return
// LLVM-LABEL: define {{.*}}void @_Z8base_refR4Base
// LLVM:       getelementptr inbounds ptr, ptr
// LLVM:       ret void
// CIR-TO-LLVM-LABEL: define {{.*}}void @_Z8base_refR4Base
// CIR-TO-LLVM:       getelementptr ptr, ptr
// CIR-TO-LLVM:       ret void

// Not most derived
void base_deref(Base *b) { ForceLoadingVTable(typeid(*b)); }
// CIR-LABEL:  cir.func {{.*}}@_Z10base_derefP4Base
// CIR:        cir.vtable.get_vptr
// CIR:        cir.return
// LLVM-LABEL: define {{.*}}void @_Z10base_derefP4Base
// LLVM:       getelementptr inbounds ptr, ptr
// LLVM:       ret void
// CIR-TO-LLVM-LABEL: define {{.*}}void @_Z10base_derefP4Base
// CIR-TO-LLVM:       getelementptr ptr, ptr
// CIR-TO-LLVM:       ret void

// Not most derived
void nonfinal_ref(NonFinal &d) { typeid(d); }
// CIR-LABEL:  cir.func {{.*}}@_Z12nonfinal_refR8NonFinal
// CIR:        cir.vtable.get_vptr
// CIR:        cir.return
// LLVM-LABEL: define {{.*}}void @_Z12nonfinal_refR8NonFinal
// LLVM:       getelementptr inbounds ptr, ptr
// LLVM:       ret void
// CIR-TO-LLVM-LABEL: define {{.*}}void @_Z12nonfinal_refR8NonFinal
// CIR-TO-LLVM:       getelementptr ptr, ptr
// CIR-TO-LLVM:       ret void

// Not most derived
void nonfinal_deref(NonFinal *d) { ForceLoadingVTable(typeid(*d)); }
// CIR-LABEL:  cir.func {{.*}}@_Z14nonfinal_derefP8NonFinal
// CIR:        cir.vtable.get_vptr
// CIR:        cir.return
// LLVM-LABEL: define {{.*}}void @_Z14nonfinal_derefP8NonFinal
// LLVM:       getelementptr inbounds ptr, ptr
// LLVM:       ret void
// CIR-TO-LLVM-LABEL: define {{.*}}void @_Z14nonfinal_derefP8NonFinal
// CIR-TO-LLVM:       getelementptr ptr, ptr
// CIR-TO-LLVM:       ret void
