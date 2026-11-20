// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fexperimental-relative-c++-abi-vtables -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefixes=CIR-AFTER,CIR-AFTER-X86 --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fexperimental-relative-c++-abi-vtables -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefixes=LLVM,LLVM-X86 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fexperimental-relative-c++-abi-vtables -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefixes=LLVM,LLVM-X86 %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -std=c++17 -fexperimental-relative-c++-abi-vtables -fclangir -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t-arm.cir 2> %t-arm-before.cir
// RUN: FileCheck --check-prefix=CIR-BEFORE --input-file=%t-arm-before.cir %s
// RUN: FileCheck --check-prefixes=CIR-AFTER,CIR-AFTER-ARM --input-file=%t-arm.cir %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -std=c++17 -fexperimental-relative-c++-abi-vtables -fclangir -emit-llvm %s -o %t-arm-cir.ll
// RUN: FileCheck --input-file=%t-arm-cir.ll --check-prefixes=LLVM,LLVM-ARM %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -std=c++17 -fexperimental-relative-c++-abi-vtables -emit-llvm %s -o %t-arm.ll
// RUN: FileCheck --input-file=%t-arm.ll --check-prefixes=LLVM,LLVM-ARM %s

namespace std {
class type_info;
}

struct A {
  virtual void f();
  virtual int g(int);
  virtual ~A();
  int a;
};

struct B : virtual A {
  virtual void h();
  int b;
};

// The index of the entry becomes the offset in bytes of the member pointer,
// with 4 bytes per entry.
int (A::*mp)(int) = &A::g;

// CIR-BEFORE: cir.global external @mp = #cir.method<vtable_index = 1>
// CIR-AFTER-X86: cir.global external @mp = #cir.const_record<{#cir.int<5> : !s64i, #cir.int<0> : !s64i}>
// CIR-AFTER-ARM: cir.global external @mp = #cir.const_record<{#cir.int<4> : !s64i, #cir.int<1> : !s64i}>
// LLVM-X86: @mp = global { i64, i64 } { i64 5, i64 0 }
// LLVM-ARM: @mp = global { i64, i64 } { i64 4, i64 1 }

// Entries that refer to a function are 32-bit offsets, read with
// llvm.load.relative at 4 bytes per entry.
int virtual_call(A *p) { return p->g(1); }

// CIR-BEFORE-LABEL: cir.func {{.*}} @_Z12virtual_callP1A
// CIR-BEFORE:         cir.vtable.get_virtual_fn %{{.*}}[1] : !cir.vptr -> !cir.ptr<!cir.func<(!cir.ptr<!rec_A>, !s32i) -> !s32i>>
// CIR-AFTER-LABEL: cir.func {{.*}} @_Z12virtual_callP1A
// CIR-AFTER:         cir.vtable.get_virtual_fn %{{.*}}[1] : !cir.vptr -> !cir.ptr<!cir.func<(!cir.ptr<!rec_A>, !s32i) -> !s32i>>
// LLVM-LABEL: define {{.*}} @_Z12virtual_callP1A
// LLVM:         call ptr @llvm.load.relative.i32(ptr %{{.*}}, i32 4)

void virtual_dtor_call(A *p) { p->~A(); }

// CIR-BEFORE-LABEL: cir.func {{.*}} @_Z17virtual_dtor_callP1A
// CIR-BEFORE:         cir.vtable.get_virtual_fn %{{.*}}[2] : !cir.vptr
// CIR-AFTER-LABEL: cir.func {{.*}} @_Z17virtual_dtor_callP1A
// CIR-AFTER:         cir.vtable.get_virtual_fn %{{.*}}[2] : !cir.vptr
// LLVM-LABEL: define {{.*}} @_Z17virtual_dtor_callP1A
// LLVM:         call ptr @llvm.load.relative.i32(ptr %{{.*}}, i32 8)

// The type_info is the entry before the address point.
const std::type_info &get_typeid(A &r) { return typeid(r); }

// CIR-BEFORE-LABEL: cir.func {{.*}} @_Z10get_typeidR1A
// CIR-BEFORE:         cir.vtable.get_type_info %{{.*}} : !cir.vptr -> !cir.ptr<!rec_std3A3Atype_info>
// CIR-AFTER-LABEL: cir.func {{.*}} @_Z10get_typeidR1A
// CIR-AFTER:         cir.vtable.get_entry %{{.*}}[-1] : !cir.vptr -> !cir.ptr<!rec_std3A3Atype_info>
// LLVM-LABEL: define {{.*}} @_Z10get_typeidR1A
// LLVM:         call ptr @llvm.load.relative.i32(ptr %{{.*}}, i32 -4)

// The offset of a virtual base is a plain 32-bit integer, not a relative
// reference. Classic codegen indexes with the 32-bit offset, while CIR extends
// it to a ptrdiff_t first.
A *vbase_cast(B *p) { return p; }

// CIR-BEFORE-LABEL: cir.func {{.*}} @_Z10vbase_castP1B
// CIR-BEFORE:         %[[OFF:.*]] = cir.vtable.get_offset %{{.*}}[-3] : !cir.vptr -> !s64i
// CIR-BEFORE:         cir.ptr_stride %{{.*}}, %[[OFF]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR-AFTER-LABEL: cir.func {{.*}} @_Z10vbase_castP1B
// CIR-AFTER:         %[[OFF:.*]] = cir.vtable.get_offset %{{.*}}[-3] : !cir.vptr -> !s64i
// CIR-AFTER:         cir.ptr_stride %{{.*}}, %[[OFF]] : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// LLVM-LABEL: define {{.*}} @_Z10vbase_castP1B
// LLVM:         %[[SLOT:.*]] = getelementptr i8, ptr %{{.*}}, i64 -12
// LLVM:         load i32, ptr %[[SLOT]], align 4
// LLVM:         getelementptr {{(inbounds )?}}i8, ptr %{{.*}}, {{i32|i64}} %{{.*}}

// A pointer to a virtual member function holds the offset of the entry in
// bytes, which is read with the 64-bit form of the intrinsic.
int call_mp(A *p, int (A::*m)(int)) { return (p->*m)(2); }

// CIR-BEFORE-LABEL: cir.func {{.*}} @_Z7call_mpP1AMS_FiiE
// CIR-BEFORE:         cir.get_method %{{.*}}, %{{.*}} : (!cir.method<!cir.func<(!cir.ptr<!rec_A>, !s32i) -> !s32i> in !rec_A>, !cir.ptr<!rec_A>)
//
// The two member function pointer ABIs differ in where they keep the flag that
// says that the function is virtual, and so in how the offset is recovered:
// - Itanium (x86_64): the flag is the low bit of 'ptr', which holds 1 + the
//   offset of the entry. The 'adj' field is the 'this' adjustment as is.
// - ARM (aarch64): the low bit of 'ptr' can't be used, as a function may be
//   only byte aligned. The flag is the low bit of 'adj', which holds twice the
//   'this' adjustment, and 'ptr' is the offset of the entry as is.
// The offset that reaches the vtable lookup is the same in both.

// CIR-AFTER-LABEL: cir.func {{.*}} @_Z7call_mpP1AMS_FiiE
// CIR-AFTER-X86:     %[[ADJ:.*]] = cir.extract_member %{{.*}}[1] : !rec_anon_struct -> !s64i
// CIR-AFTER-X86:     cir.ptr_stride %{{.*}}, %[[ADJ]] : (!cir.ptr<!void>, !s64i) -> !cir.ptr<!void>
// CIR-AFTER-X86:     %[[PTR:.*]] = cir.extract_member %{{.*}}[0] : !rec_anon_struct -> !s64i
// CIR-AFTER-X86:     %[[BIT:.*]] = cir.and %[[PTR]], %[[ONE:.*]] : !s64i
// CIR-AFTER-X86:     cir.cmp eq %[[BIT]], %[[ONE]] : !s64i
// CIR-AFTER-ARM:     %[[ADJ:.*]] = cir.extract_member %{{.*}}[1] : !rec_anon_struct -> !s64i
// CIR-AFTER-ARM:     %[[ADJ_SHIFT:.*]] = cir.shift(right, %[[ADJ]] : !s64i, %[[ONE:.*]] : !s64i) -> !s64i
// CIR-AFTER-ARM:     cir.ptr_stride %{{.*}}, %[[ADJ_SHIFT]] : (!cir.ptr<!void>, !s64i) -> !cir.ptr<!void>
// CIR-AFTER-ARM:     %[[PTR:.*]] = cir.extract_member %{{.*}}[0] : !rec_anon_struct -> !s64i
// CIR-AFTER-ARM:     %[[BIT:.*]] = cir.and %[[ADJ]], %[[ONE]] : !s64i
// CIR-AFTER-ARM:     cir.cmp eq %[[BIT]], %[[ONE]] : !s64i
// CIR-AFTER:         cir.vtable.get_vptr %{{.*}} : !cir.ptr<!void> -> !cir.ptr<!cir.vptr>
// CIR-AFTER:         %[[VPTR:.*]] = cir.load {{.*}} : !cir.ptr<!cir.vptr>, !cir.vptr
// CIR-AFTER-X86:     %[[OFFSET:.*]] = cir.sub %[[PTR]], %[[ONE]] : !s64i
// CIR-AFTER-X86:     cir.vtable.get_virtual_fn_by_offset %[[VPTR]][%[[OFFSET]] : !s64i] : !cir.vptr -> !cir.ptr<!cir.func<(!cir.ptr<!void>, !s32i) -> !s32i>>
// CIR-AFTER-ARM:     cir.vtable.get_virtual_fn_by_offset %[[VPTR]][%[[PTR]] : !s64i] : !cir.vptr -> !cir.ptr<!cir.func<(!cir.ptr<!void>, !s32i) -> !s32i>>

// LLVM-LABEL: define {{.*}} @_Z7call_mpP1AMS_FiiE
// LLVM-X86:     %[[ADJ:.*]] = extractvalue { i64, i64 } %{{.*}}, 1
// LLVM-X86:     getelementptr {{(inbounds )?}}i8, ptr %{{.*}}, i64 %[[ADJ]]
// LLVM-X86:     %[[PTR:.*]] = extractvalue { i64, i64 } %{{.*}}, 0
// LLVM-X86:     and i64 %[[PTR]], 1
// LLVM-X86:     %[[OFFSET:.*]] = sub i64 %[[PTR]], 1
// LLVM-X86:     call ptr @llvm.load.relative.i64(ptr %{{.*}}, i64 %[[OFFSET]])
// LLVM-ARM:     %[[ADJ:.*]] = extractvalue { i64, i64 } %{{.*}}, 1
// LLVM-ARM:     %[[ADJ_SHIFT:.*]] = ashr i64 %[[ADJ]], 1
// LLVM-ARM:     getelementptr {{(inbounds )?}}i8, ptr %{{.*}}, i64 %[[ADJ_SHIFT]]
// LLVM-ARM:     %[[PTR:.*]] = extractvalue { i64, i64 } %{{.*}}, 0
// LLVM-ARM:     and i64 %[[ADJ]], 1
// LLVM-ARM:     call ptr @llvm.load.relative.i64(ptr %{{.*}}, i64 %[[PTR]])
