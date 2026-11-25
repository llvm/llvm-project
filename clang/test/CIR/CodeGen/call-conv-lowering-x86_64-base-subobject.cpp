// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Base is trivial but not standard-layout, so its base subobject is a
// distinct packed CIR record: five bytes, which is not a multiple of the
// int member's ABI alignment. alignas(8) gives that subobject language
// alignment 8. The packed field places it at offset 4, which is not a
// multiple of 8, so the x86-64 ABI passes Holder in memory.

struct alignas(8) Base {
  int i;
private:
  char c;
};

struct Holder {
  char prefix[4];
  [[no_unique_address]] Base b __attribute__((packed));
};

void take(Holder h) {}

// CIR-DAG: !rec_Base2Ebase = !cir.struct<"Base.base" packed {data !s32i, data !s8i}>
// CIR-DAG: !rec_Holder = !cir.struct<"Holder" {data !cir.array<!s8i x 4>, data !rec_Base2Ebase, pad !cir.array<!u8i x 3>}>
// CIR-DAG: Base.base = #cir.record_layout<arg_passing_kind = can_pass_in_regs, has_trivial_dtor = true, record_align = 8, unadjusted_align = 4>

// CIR-LABEL: cir.func {{.*}} @_Z4take6Holder(
// CIR-SAME:    !cir.ptr<!rec_Holder> {llvm.align = 8 : i64, llvm.byval = !rec_Holder, llvm.noundef}

// LLVM-LABEL: define dso_local void @_Z4take6Holder(ptr noundef byval(%struct.Holder) align 8 %{{.*}})
