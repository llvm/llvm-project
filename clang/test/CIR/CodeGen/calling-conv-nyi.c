// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -DVECTORCALL %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=VECTORCALL
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -DMS_ABI %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=MS_ABI
// RUN: not %clang_cc1 -triple i386-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -DFASTCALL %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=FASTCALL
// RUN: not %clang_cc1 -triple i386-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -x c++ -DFASTCALL %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=FASTCALL
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -x c++ -DVECTORCALL %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=VECTORCALL

// Target-specific calling conventions are not representable in CIR yet. Make
// sure they are diagnosed instead of silently falling back to the C calling
// convention.

#if defined(VECTORCALL)
#define CC __attribute__((vectorcall))
#elif defined(MS_ABI)
#define CC __attribute__((ms_abi))
#elif defined(FASTCALL)
#define CC __attribute__((fastcall))
#endif

int CC callee(int x);

int caller(int x) { return callee(x); }

// VECTORCALL: ClangIR code gen Not Yet Implemented: calling convention: vectorcall
// MS_ABI: ClangIR code gen Not Yet Implemented: calling convention: ms_abi
// FASTCALL: ClangIR code gen Not Yet Implemented: calling convention: fastcall
