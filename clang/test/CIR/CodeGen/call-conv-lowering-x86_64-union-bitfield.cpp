// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct E {};

union BitEmptySpan {
  E e[4];
  int x : 8;
};

union NoRegs {
  int x : 3;
  NoRegs() {}
  ~NoRegs() {}
};

// CIR-DAG: !rec_BitEmptySpan = !cir.union<"BitEmptySpan" {data !cir.array<!rec_E x 4>, bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 8>]>}>
// CIR-DAG: !rec_NoRegs = !cir.union<"NoRegs" {bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 3>]>}, padding = {!cir.array<!u8i x 3>}>

// The empty-record array covers the union but supplies no data, so the
// bit-field has to cover it too, which it does through its declared type.
void take_bit_empty_span(BitEmptySpan u) {}
// CIR: cir.func{{.*}} @_Z19take_bit_empty_span12BitEmptySpan(%arg0: !u32i loc
// LLVM: define{{.*}} void @_Z19take_bit_empty_span12BitEmptySpan(i32 %{{[^,)]+}})

// A union that cannot pass in registers is returned through an sret slot, so
// the declared extent decides only whether it can be classified at all.
NoRegs ret_no_regs() { return NoRegs(); }
// CIR: cir.func{{.*}} @_Z11ret_no_regsv(%arg0: !cir.ptr<!rec_NoRegs> {llvm.align = 4 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_NoRegs, llvm.writable} loc
// LLVM: define{{.*}} void @_Z11ret_no_regsv(ptr dead_on_unwind noalias writable sret(%union.NoRegs) align 4 %{{[^,)]+}})
