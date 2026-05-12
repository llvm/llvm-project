// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// `cir.set_bitfield` and `cir.get_bitfield` produce integer values only,
// but a bit-field whose declared type is `bool` (e.g. `bool flag : 1;`,
// common in libcxx's `std::format` parsing state) is naturally typed as
// `!cir.bool` at the CIRGen layer.  CIRGen widens `bool` to the storage's
// integer type for the op call and narrows back to `bool` with
// `int_to_bool` for callers that consume the result.

struct B {
  bool flag : 1;
  bool other : 1;
};

// Store: object expression is unused, so the int_to_bool on the stored
// value gets DCE'd by the `Pure` trait on `cir.cast`; we just want to
// see the bool source widened with `bool_to_int` and a `!u8i`-typed
// `cir.set_bitfield`.

void store_bool_bitfield(B *b) {
  b->flag = true;
}

// CIR-LABEL: cir.func{{.*}} @_Z19store_bool_bitfieldP1B
// CIR:         %[[TRUE:.+]] = cir.const #true
// CIR:         %[[WIDEN:.+]] = cir.cast bool_to_int %[[TRUE]] : !cir.bool -> !u8i
// CIR:         cir.set_bitfield{{.*}}(#{{.+}}, %{{.+}} : !cir.ptr<!u8i>, %[[WIDEN]] : !u8i) -> !u8i

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

// Assignment-expression value used: the store result must be narrowed
// back to bool with int_to_bool so the caller sees a `!cir.bool`.

bool store_bool_bitfield_used(B *b, bool v) {
  return b->flag = v;
}

// CIR-LABEL: cir.func{{.*}} @_Z24store_bool_bitfield_usedP1Bb
// CIR:         %[[V:.+]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:         %[[VINT:.+]] = cir.cast bool_to_int %[[V]] : !cir.bool -> !u8i
// CIR:         %[[STORED:.+]] = cir.set_bitfield {{.*}} %[[VINT]] : !u8i) -> !u8i
// CIR:         %[[BACK:.+]] = cir.cast int_to_bool %[[STORED]] : !u8i -> !cir.bool

// LLVM-LABEL: define {{.*}}i1 @_Z24store_bool_bitfield_usedP1Bb
// LLVM:         zext i1 %{{.+}} to i8
// LLVM:         store i8 %{{.+}}, ptr %{{.+}}
// LLVM:         icmp ne i8 %{{.+}}, 0

// OGCG-LABEL: define {{.*}}i1 @_Z24store_bool_bitfield_usedP1Bb
// OGCG:         zext i1 %{{.+}} to i8
// OGCG:         store i8 %{{.+}}, ptr %{{.+}}
// OGCG:         icmp ne i8 %{{.+}}, 0

// Load: read a bool bitfield.  The op produces an integer, which CIRGen
// then narrows to `!cir.bool` with `int_to_bool`.

bool load_bool_bitfield(B *b) {
  return b->flag;
}

// CIR-LABEL: cir.func{{.*}} @_Z18load_bool_bitfieldP1B
// CIR:         %[[INT:.+]] = cir.get_bitfield{{.*}}(#{{.+}}, %{{.+}} : !cir.ptr<!u8i>) -> !u8i
// CIR:         %[[BOOL:.+]] = cir.cast int_to_bool %[[INT]] : !u8i -> !cir.bool

// LLVM-LABEL: define {{.*}}i1 @_Z18load_bool_bitfieldP1B
// LLVM:         load i8, ptr %{{.+}}
// LLVM:         and i8 %{{.+}}, 1
// LLVM:         icmp ne i8 %{{.+}}, 0

// OGCG-LABEL: define {{.*}}i1 @_Z18load_bool_bitfieldP1B
// OGCG:         load i8, ptr %{{.+}}
// OGCG:         and i8 %{{.+}}, 1
// OGCG:         trunc i8 %{{.+}} to i1

// Compound assignment to a `bool` bit-field (e.g. `state.flags |= v;`,
// common in libcxx `<format>` parsing state).  The read-modify-write
// path goes through both `emitLoadOfBitfieldLValue` and
// `emitStoreThroughBitfieldLValue` and so exercises both halves of
// the fix.

void compound_or_bool_bitfield(B *b, bool v) {
  b->flag |= v;
}

// CIR-LABEL: cir.func{{.*}} @_Z25compound_or_bool_bitfieldP1Bb
// CIR:         cir.get_bitfield{{.*}}-> !u8i
// CIR:         cir.cast int_to_bool %{{.+}} : !u8i -> !cir.bool
// CIR:         %[[WIDEN:.+]] = cir.cast bool_to_int %{{.+}} : !cir.bool -> !u8i
// CIR:         cir.set_bitfield{{.*}}, %[[WIDEN]] : !u8i) -> !u8i

// Bit-field-to-bit-field copy (e.g. `state.a = state.b;` between two
// `bool` bit-fields in the same record).

void copy_bool_bitfield(B *b) {
  b->flag = b->other;
}

// CIR-LABEL: cir.func{{.*}} @_Z18copy_bool_bitfieldP1B
// CIR:         cir.get_bitfield{{.*}}-> !u8i
// CIR:         cir.cast int_to_bool %{{.+}} : !u8i -> !cir.bool
// CIR:         %[[WIDEN:.+]] = cir.cast bool_to_int %{{.+}} : !cir.bool -> !u8i
// CIR:         cir.set_bitfield{{.*}}, %[[WIDEN]] : !u8i) -> !u8i

// Mix of bool and int bitfields in the same storage word — regression
// check that ordinary int-typed bitfields keep their previous code
// shape (no extra bool_to_int / int_to_bool).

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
