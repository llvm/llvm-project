// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -I%S/../Inputs -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -I%S/../Inputs -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

#include "std-cxx.h"

// CIR-LABEL:  @_Z3fooPKc
// LLVM-LABEL: @_Z3fooPKc

void foo(const char* path) {
  std::string str = path;
  str = path;
  str = path;
}

// CIR: cir.try synthetic cleanup {
// CIR:   cir.call exception @_ZNSbIcEC1EPKcRKNS_9AllocatorE({{.*}}, {{.*}}, {{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!s8i>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E3A3AAllocator>) -> () cleanup {
// CIR:     cir.call @_ZNSbIcED1Ev({{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> ()
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.yield
// CIR: } catch [#cir.unwind {
// CIR:   cir.resume
// CIR: }]
// CIR: cir.try synthetic cleanup {
// CIR:   {{.*}} = cir.call exception @_ZNSbIcEaSERKS_({{.*}}, {{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> !cir.ptr<!ty_std3A3Abasic_string3Cchar3E> cleanup {
// CIR:     cir.call @_ZNSbIcED1Ev({{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> ()
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.store {{.*}}, {{.*}} : !cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>>
// CIR:   cir.yield
// CIR: } catch [#cir.unwind {
// CIR:   cir.resume
// CIR: }]
// CIR: {{.*}} = cir.load {{.*}} : !cir.ptr<!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E>
// CIR: cir.call @_ZNSbIcED1Ev({{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> ()
// CIR: cir.try synthetic cleanup {
// CIR:   cir.call exception @_ZNSbIcEC1EPKcRKNS_9AllocatorE({{.*}}, {{.*}}, {{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!s8i>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E3A3AAllocator>) -> ()
// CIR:   cir.yield
// CIR: } catch [#cir.unwind {
// CIR:   cir.resume
// CIR: }]
// CIR: cir.try synthetic cleanup {
// CIR:   {{.*}} = cir.call exception @_ZNSbIcEaSERKS_({{.*}}, {{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> !cir.ptr<!ty_std3A3Abasic_string3Cchar3E> cleanup {
// CIR:     cir.call @_ZNSbIcED1Ev({{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> ()
// CIR:     cir.call @_ZNSbIcED1Ev({{.*}}) : (!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>) -> ()
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.store {{.*}}, {{.*}} : !cir.ptr<!ty_std3A3Abasic_string3Cchar3E>, !cir.ptr<!cir.ptr<!ty_std3A3Abasic_string3Cchar3E>>
// CIR:   cir.yield
// CIR: } catch [#cir.unwind {
// CIR:   cir.resume
// CIR: }]

// LLVM:  invoke void @_ZNSbIcEC1EPKcRKNS_9AllocatorE(ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// LLVM:           to label {{.*}} unwind label %[[B18:.*]]
// LLVM: [[B18]]
// LLVM:   call void @_ZNSbIcED1Ev(ptr {{.*}})
// LLVM:   br label %[[B22:.*]]
// LLVM: [[B22]]
// LLVM:   resume { ptr, i32 } {{.*}}
// LLVM: {{.*}}:
// LLVM:   {{.*}} = invoke ptr @_ZNSbIcEaSERKS_(ptr {{.*}}, ptr {{.*}})
// LLVM:           to label {{.*}} unwind label %[[B31:.*]]
// LLVM: [[B31]]
// LLVM:   call void @_ZNSbIcED1Ev(ptr {{.*}})
// LLVM:   br label %[[B35:.*]]
// LLVM: [[B35]]
// LLVM:   resume { ptr, i32 } {{.*}}
// LLVM: {{.*}}:
// LLVM:   call void @_ZNSbIcED1Ev(ptr {{.*}})
// LLVM:   br label {{.*}}
// LLVM: {{.*}}:
// LLVM:   invoke void @_ZNSbIcEC1EPKcRKNS_9AllocatorE(ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// LLVM:           to label {{.*}} unwind label %[[B46:.*]]
// LLVM: [[B46]]
// LLVM:   br label %[[B50:.*]]
// LLVM: [[B50]]
// LLVM:   resume { ptr, i32 } {{.*}}
// LLVM: {{.*}}:
// LLVM:   {{.*}} = invoke ptr @_ZNSbIcEaSERKS_(ptr {{.*}}, ptr {{.*}})
// LLVM:           to label {{.*}} unwind label %[[B59:.*]]
// LLVM: [[B59]]
// LLVM:   call void @_ZNSbIcED1Ev(ptr {{.*}})
// LLVM:   call void @_ZNSbIcED1Ev(ptr {{.*}})
// LLVM:   br label %[[B63:.*]]
// LLVM: [[B63]]
// LLVM:   resume { ptr, i32 } {{.*}}
