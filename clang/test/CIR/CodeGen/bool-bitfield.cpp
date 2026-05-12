// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct B {
  bool flag : 1;
  bool other : 1;
};

void store_bool_bitfield(B *b) {
  b->flag = true;
}

// CIR-LABEL: cir.func{{.*}} @_Z19store_bool_bitfieldP1B
// CIR:         %[[TRUE:.+]] = cir.const #true
// CIR:         cir.set_bitfield{{.*}}(#{{.+}}, %{{.+}} : !cir.ptr<!u8i>, %[[TRUE]] : !cir.bool) -> !cir.bool

// LLVM-LABEL: define {{.*}} void @_Z19store_bool_bitfieldP1B
// LLVM:         load ptr, ptr %{{.+}}
// LLVM:         load i8, ptr %{{.+}}
// LLVM:         %{{.+}} = and i8 %{{.+}}, -2
// LLVM:         %{{.+}} = or {{.*}}i8 %{{.+}}, 1
// LLVM:         store i8 %{{.+}}, ptr %{{.+}}

// OGCG-LABEL: define {{.*}} void @_Z19store_bool_bitfieldP1B
// OGCG:         load ptr, ptr %{{.+}}
// OGCG:         load i8, ptr %{{.+}}
// OGCG:         %{{.+}} = and i8 %{{.+}}, -2
// OGCG:         %{{.+}} = or {{.*}}i8 %{{.+}}, 1
// OGCG:         store i8 %{{.+}}, ptr %{{.+}}

bool store_bool_bitfield_used(B *b, bool v) {
  return b->flag = v;
}

// CIR-LABEL: cir.func{{.*}} @_Z24store_bool_bitfield_usedP1Bb
// CIR:         %[[V:.+]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:         cir.set_bitfield{{.*}}, %[[V]] : !cir.bool) -> !cir.bool

// LLVM-LABEL: define {{.*}}i1 @_Z24store_bool_bitfield_usedP1Bb
// LLVM:         zext i1 %{{.+}} to i8
// LLVM:         store i8 %{{.+}}, ptr %{{.+}}

// OGCG-LABEL: define {{.*}}i1 @_Z24store_bool_bitfield_usedP1Bb
// OGCG:         zext i1 %{{.+}} to i8
// OGCG:         store i8 %{{.+}}, ptr %{{.+}}

bool load_bool_bitfield(B *b) {
  return b->flag;
}

// CIR-LABEL: cir.func{{.*}} @_Z18load_bool_bitfieldP1B
// CIR:         cir.get_bitfield{{.*}}(#{{.+}}, %{{.+}} : !cir.ptr<!u8i>) -> !cir.bool

// LLVM-LABEL: define {{.*}}i1 @_Z18load_bool_bitfieldP1B
// LLVM:         load i8, ptr %{{.+}}
// LLVM:         and i8 %{{.+}}, 1
// LLVM:         trunc i8 %{{.+}} to i1

// OGCG-LABEL: define {{.*}}i1 @_Z18load_bool_bitfieldP1B
// OGCG:         load i8, ptr %{{.+}}
// OGCG:         and i8 %{{.+}}, 1
// OGCG:         trunc i8 %{{.+}} to i1

void compound_or_bool_bitfield(B *b, bool v) {
  b->flag |= v;
}

// CIR-LABEL: cir.func{{.*}} @_Z25compound_or_bool_bitfieldP1Bb
// CIR:         cir.get_bitfield{{.*}}-> !cir.bool
// CIR:         cir.set_bitfield{{.*}} : !cir.bool) -> !cir.bool

void copy_bool_bitfield(B *b) {
  b->flag = b->other;
}

// CIR-LABEL: cir.func{{.*}} @_Z18copy_bool_bitfieldP1B
// CIR:         %[[OTHER:.+]] = cir.get_bitfield{{.*}}-> !cir.bool
// CIR:         cir.set_bitfield{{.*}}, %[[OTHER]] : !cir.bool) -> !cir.bool

struct M {
  bool b : 1;
  int  n : 7;
};

int load_int_bitfield(M *m) {
  return m->n;
}

// CIR-LABEL: cir.func{{.*}} @_Z17load_int_bitfieldP1M
// CIR:         cir.get_bitfield{{.*}}(#{{.+}}, %{{.+}} : !cir.ptr<!u8i>) -> !s32i
// CIR-NOT:     cir.cast int_to_bool

void store_int_bitfield(M *m) {
  m->n = 5;
}

// CIR-LABEL: cir.func{{.*}} @_Z18store_int_bitfieldP1M
// CIR-NOT:     cir.cast bool_to_int
// CIR:         cir.set_bitfield{{.*}}(#{{.+}}, %{{.+}} : !cir.ptr<!u8i>, %{{.+}} : !s32i) -> !s32i
