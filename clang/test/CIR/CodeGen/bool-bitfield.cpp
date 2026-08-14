// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

struct B {
  bool flag : 1;
  bool other : 1;
};

void store_bool_bitfield(B *b) {
  b->flag = true;
}

// CIR-LABEL: cir.func{{.*}} @_Z19store_bool_bitfieldP1B
// CIR:         %[[TRUE:.+]] = cir.const #true
// CIR:         %[[B_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>
// CIR:         %[[FLAG_PTR:.+]] = cir.get_member %[[B_PTR]][0] {name = "flag"} : !cir.ptr<!rec_B> -> !cir.ptr<!u8i>
// CIR:         cir.set_bitfield align(1) (#bfi_flag, %[[FLAG_PTR]] : !cir.ptr<!u8i>, %[[TRUE]] : !cir.bool) -> !cir.bool

// LLVM-LABEL: define {{.*}} void @_Z19store_bool_bitfieldP1B
// LLVM:         %[[OLD:.+]] = load i8, ptr %{{.+}}
// LLVM:         %[[CLEARED:.+]] = and i8 %[[OLD]], -2
// LLVM:         %[[NEW:.+]] = or {{.*}}i8 %[[CLEARED]], 1
// LLVM:         store i8 %[[NEW]], ptr %{{.+}}

bool store_bool_bitfield_used(B *b, bool v) {
  return b->flag = v;
}

// CIR-LABEL: cir.func{{.*}} @_Z24store_bool_bitfield_usedP1Bb
// CIR:         %[[V:.+]] = cir.load align(1) %{{.+}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:         %[[B_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>
// CIR:         %[[FLAG_PTR:.+]] = cir.get_member %[[B_PTR]][0] {name = "flag"} : !cir.ptr<!rec_B> -> !cir.ptr<!u8i>
// CIR:         cir.set_bitfield align(1) (#bfi_flag, %[[FLAG_PTR]] : !cir.ptr<!u8i>, %[[V]] : !cir.bool) -> !cir.bool

// LLVM-LABEL: define {{.*}}i1 @_Z24store_bool_bitfield_usedP1Bb
// LLVM:         %{{.+}} = zext i1 %{{.+}} to i8
// LLVM:         %[[CLEARED:.+]] = and i8 %{{.+}}, -2
// LLVM:         %[[NEW:.+]] = or i8 %[[CLEARED]], %{{.+}}
// LLVM:         store i8 %[[NEW]], ptr %{{.+}}

bool load_bool_bitfield(B *b) {
  return b->flag;
}

// CIR-LABEL: cir.func{{.*}} @_Z18load_bool_bitfieldP1B
// CIR:         %[[B_PTR:.+]] = cir.load align(8) %{{.+}} : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>
// CIR:         %[[FLAG_PTR:.+]] = cir.get_member %[[B_PTR]][0] {name = "flag"} : !cir.ptr<!rec_B> -> !cir.ptr<!u8i>
// CIR:         %[[FLAG:.+]] = cir.get_bitfield align(1) (#bfi_flag, %[[FLAG_PTR]] : !cir.ptr<!u8i>) -> !cir.bool

// LLVM-LABEL: define {{.*}}i1 @_Z18load_bool_bitfieldP1B
// LLVM:         %[[OLD:.+]] = load i8, ptr %{{.+}}
// LLVM:         %[[MASKED:.+]] = and i8 %[[OLD]], 1
// LLVM:         %{{.+}} = trunc i8 %[[MASKED]] to i1

void compound_or_bool_bitfield(B *b, bool v) {
  b->flag |= v;
}

// CIR-LABEL: cir.func{{.*}} @_Z25compound_or_bool_bitfieldP1Bb
// CIR:         %[[V:.+]] = cir.load align(1) %{{.+}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:         %[[V_I32:.+]] = cir.cast bool_to_int %[[V]] : !cir.bool -> !s32i
// CIR:         %[[FLAG_PTR:.+]] = cir.get_member %{{.+}}[0] {name = "flag"} : !cir.ptr<!rec_B> -> !cir.ptr<!u8i>
// CIR:         %[[OLD:.+]] = cir.get_bitfield align(1) (#bfi_flag, %[[FLAG_PTR]] : !cir.ptr<!u8i>) -> !cir.bool
// CIR:         %[[OLD_I32:.+]] = cir.cast bool_to_int %[[OLD]] : !cir.bool -> !s32i
// CIR:         %[[OR:.+]] = cir.or %[[OLD_I32]], %[[V_I32]] : !s32i
// CIR:         %[[NEW:.+]] = cir.cast int_to_bool %[[OR]] : !s32i -> !cir.bool
// CIR:         cir.set_bitfield align(1) (#bfi_flag, %[[FLAG_PTR]] : !cir.ptr<!u8i>, %[[NEW]] : !cir.bool) -> !cir.bool

// LLVM-LABEL: define {{.*}} void @_Z25compound_or_bool_bitfieldP1Bb
// LLVM:         %[[CLEARED:.+]] = and i8 %{{.+}}, -2
// LLVM:         %[[NEW:.+]] = or i8 %[[CLEARED]], %{{.+}}
// LLVM:         store i8 %[[NEW]], ptr %{{.+}}

void copy_bool_bitfield(B *b) {
  b->flag = b->other;
}

// CIR-LABEL: cir.func{{.*}} @_Z18copy_bool_bitfieldP1B
// CIR:         %[[OTHER_PTR:.+]] = cir.get_member %{{.+}}[0] {name = "other"} : !cir.ptr<!rec_B> -> !cir.ptr<!u8i>
// CIR:         %[[OTHER:.+]] = cir.get_bitfield align(1) (#bfi_other, %[[OTHER_PTR]] : !cir.ptr<!u8i>) -> !cir.bool
// CIR:         %[[FLAG_PTR:.+]] = cir.get_member %{{.+}}[0] {name = "flag"} : !cir.ptr<!rec_B> -> !cir.ptr<!u8i>
// CIR:         cir.set_bitfield align(1) (#bfi_flag, %[[FLAG_PTR]] : !cir.ptr<!u8i>, %[[OTHER]] : !cir.bool) -> !cir.bool

// LLVM-LABEL: define {{.*}} void @_Z18copy_bool_bitfieldP1B
// LLVM:         %[[OTHER_BYTE:.+]] = load i8, ptr %{{.+}}
// LLVM:         %[[OTHER_SHIFTED:.+]] = lshr i8 %[[OTHER_BYTE]], 1
// LLVM:         %[[OTHER_BIT:.+]] = and i8 %[[OTHER_SHIFTED]], 1
// LLVM:         %{{.+}} = trunc i8 %[[OTHER_BIT]] to i1
// LLVM:         %[[FLAG_BYTE:.+]] = load i8, ptr %{{.+}}
// LLVM:         %[[FLAG_CLEARED:.+]] = and i8 %[[FLAG_BYTE]], -2
// LLVM:         %[[NEW:.+]] = or i8 %[[FLAG_CLEARED]], %{{.+}}
// LLVM:         store i8 %[[NEW]], ptr %{{.+}}

struct M {
  bool b : 1;
  int  n : 7;
};

int load_int_bitfield(M *m) {
  return m->n;
}

// CIR-LABEL: cir.func{{.*}} @_Z17load_int_bitfieldP1M
// CIR:         %[[N_PTR:.+]] = cir.get_member %{{.+}}[0] {name = "n"} : !cir.ptr<!rec_M> -> !cir.ptr<!u8i>
// CIR:         %[[N:.+]] = cir.get_bitfield align(4) (#bfi_n, %[[N_PTR]] : !cir.ptr<!u8i>) -> !s32i
// CIR-NOT:     cir.cast int_to_bool

// LLVM-LABEL: define {{.*}}i32 @_Z17load_int_bitfieldP1M
// LLVM:         %[[BYTE:.+]] = load i8, ptr %{{.+}}
// LLVM:         %[[SHIFTED:.+]] = ashr i8 %[[BYTE]], 1
// LLVM:         %{{.+}} = sext i8 %[[SHIFTED]] to i32

void store_int_bitfield(M *m) {
  m->n = 5;
}

// CIR-LABEL: cir.func{{.*}} @_Z18store_int_bitfieldP1M
// CIR:         %[[FIVE:.+]] = cir.const #cir.int<5> : !s32i
// CIR:         %[[N_PTR:.+]] = cir.get_member %{{.+}}[0] {name = "n"} : !cir.ptr<!rec_M> -> !cir.ptr<!u8i>
// CIR-NOT:     cir.cast bool_to_int
// CIR:         cir.set_bitfield align(4) (#bfi_n, %[[N_PTR]] : !cir.ptr<!u8i>, %[[FIVE]] : !s32i) -> !s32i

// LLVM-LABEL: define {{.*}} void @_Z18store_int_bitfieldP1M
// LLVM:         %[[OLD:.+]] = load i8, ptr %{{.+}}
// LLVM:         %[[CLEARED:.+]] = and i8 %[[OLD]], 1
// LLVM:         %[[NEW:.+]] = or i8 %[[CLEARED]], 10
// LLVM:         store i8 %[[NEW]], ptr %{{.+}}
