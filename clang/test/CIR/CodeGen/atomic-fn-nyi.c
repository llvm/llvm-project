// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir -DARG %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=NYI
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir -DRET %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=NYI
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir -DSTRUCT %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=NYI
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir -DARRAY %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=NYI
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir -x c++ -DBASE %s -o %t.cir 2>&1 | FileCheck %s \
// RUN:   --check-prefix=NYI
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -DPTR %s -o %t.cir
// RUN: FileCheck --check-prefix=PTR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -DDECAY %s -o %t.cir
// RUN: FileCheck --check-prefix=DECAY --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir \
// RUN:   -x c++ -DREF %s -o %t.cir
// RUN: FileCheck --check-prefix=REF --input-file=%t.cir %s

// NYI: ClangIR code gen Not Yet Implemented: passing or returning atomic types

#ifdef ARG
void atomic_arg(_Atomic int x) {}
#endif

#ifdef RET
_Atomic int atomic_ret(void) { return 0; }
#endif

#ifdef STRUCT
struct HasAtomic {
  _Atomic int x;
};
void atomic_struct(struct HasAtomic s) {}
#endif

#ifdef ARRAY
struct HasAtomicArray {
  _Atomic int a[2];
};
void atomic_array(struct HasAtomicArray s) {}
#endif

#ifdef BASE
struct AtomicBase {
  _Atomic int x;
};
struct AtomicDerived : AtomicBase {};
void atomic_base(AtomicDerived d) {}
#endif

#ifdef PTR
void atomic_ptr(_Atomic int *p) {}
struct HasAtomicPtr {
  _Atomic int *p;
};
void atomic_ptr_in_struct(struct HasAtomicPtr s) {}
// PTR-LABEL: @atomic_ptr
// PTR-LABEL: @atomic_ptr_in_struct
#endif

#ifdef DECAY
// Array parameters decay to pointers, so this is not an atomic by-value ABI
// type.
void atomic_array_param(_Atomic int a[2]) {}
// DECAY-LABEL: @atomic_array_param
#endif

#ifdef REF
void atomic_ref(_Atomic int &x) {}
// REF-LABEL: @_Z10atomic_refRU7_Atomici
#endif
