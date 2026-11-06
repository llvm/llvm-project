// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -Wno-return-type %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-C
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -Wno-return-type -x c++ %s -o %t-cxx.cir
// RUN: FileCheck --input-file=%t-cxx.cir %s -check-prefixes=CIR,CIR-CXX-O0
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -Wno-return-type -x c++ -O1 %s -o %t-cxx-O1.cir
// RUN: FileCheck --input-file=%t-cxx-O1.cir %s -check-prefixes=CIR,CIR-CXX-O1

// The terminator CIRGen synthesizes when control flows off the end of a
// function body is marked `implicit`; terminators written by the user are not.

#ifdef __cplusplus
extern "C" {
#endif

void void_falloff(void) {}

// CIR-LABEL: cir.func {{.*}} @void_falloff()
// CIR:         cir.return implicit

int nonvoid_falloff(int x) {
  if (x)
    return 1;
}

// CIR-LABEL:  cir.func {{.*}} @nonvoid_falloff(
// CIR:            cir.return %{{.*}} : !s32i
// CIR-C:        %[[RV:.*]] = cir.load %{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR-C-NEXT:   cir.return implicit %[[RV]] : !s32i
// CIR-CXX-O0:   cir.trap implicit
// CIR-CXX-O1:   cir.unreachable implicit

int explicit_return(void) { return 0; }

// CIR-LABEL: cir.func {{.*}} @explicit_return()
// CIR:         cir.return %{{.*}} : !s32i
// CIR-NOT:     implicit
// CIR:       }

int user_trap(void) { __builtin_trap(); }

// CIR-LABEL: cir.func {{.*}} @user_trap()
// CIR:         cir.trap
// CIR-NOT:     implicit
// CIR:       }

int user_unreachable(void) { __builtin_unreachable(); }

// CIR-LABEL: cir.func {{.*}} @user_unreachable()
// CIR:         cir.unreachable
// CIR-NOT:     implicit
// CIR:       }

#ifdef __cplusplus
}
#endif
