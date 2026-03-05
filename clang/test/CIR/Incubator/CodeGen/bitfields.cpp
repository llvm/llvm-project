// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct __long {
  struct __attribute__((__packed__)) {
      unsigned __is_long_ : 1;
      unsigned __cap_ : sizeof(unsigned) * 8 - 1;
  };
  unsigned __size_;
  unsigned *__data_;
};
// CHECK-DAG: !rec___long = !cir.record<struct "__long" {!rec_anon2E0, !u32i, !cir.ptr<!u32i>}>

void m() {
  __long l;
}

typedef struct {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e : 15;
  unsigned f; // type other than int above, not a bitfield
} S;
// CHECK-DAG: !rec_S = !cir.record<struct "S" {!u64i, !u16i, !u32i}>
// CHECK-DAG: #bfi_a = #cir.bitfield_info<name = "a", storage_type = !u64i, size = 4, offset = 0, is_signed = true>
typedef struct {
  int a : 3;  // one bitfield with size < 8
  unsigned b;
} T;
// CHECK-DAG: !rec_T = !cir.record<struct "T" {!u8i, !u32i} #cir.record.decl.ast>
// CHECK-DAG: #bfi_a1 = #cir.bitfield_info<name = "a", storage_type = !u8i, size = 3, offset = 0, is_signed = true>
// CHECK-DAG: !rec_anon2E0 = !cir.record<struct "anon.0" {!u32i} #cir.record.decl.ast>

// CHECK: cir.func {{.*}} @_Z11store_field
// CHECK:   [[TMP0:%.*]] = cir.alloca !rec_S, !cir.ptr<!rec_S>
// CHECK:   [[TMP1:%.*]] = cir.const #cir.int<3> : !s32i
// CHECK:   [[TMP2:%.*]] = cir.get_member [[TMP0]][0] {name = "a"} : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CHECK:   cir.set_bitfield align(4) (#bfi_a, [[TMP2]] : !cir.ptr<!u64i>, [[TMP1]] : !s32i)
void store_field() {
  S s;
  s.a = 3;
}

// CHECK: cir.func {{.*}} @_Z10load_field
// CHECK:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["s", init, const]
// CHECK:   [[TMP1:%.*]] = cir.load [[TMP0]] : !cir.ptr<!cir.ptr<!rec_S>>, !cir.ptr<!rec_S>
// CHECK:   [[TMP2:%.*]] = cir.get_member [[TMP1]][0] {name = "d"} : !cir.ptr<!rec_S> -> !cir.ptr<!u64i>
// CHECK:   [[TMP3:%.*]] = cir.get_bitfield align(4) (#bfi_d, [[TMP2]] : !cir.ptr<!u64i>) -> !s32i
int load_field(S& s) {
  return s.d;
}

// CHECK: cir.func {{.*}} @_Z17load_non_bitfield
// CHECK:   cir.get_member {{%.}}[2] {name = "f"} : !cir.ptr<!rec_S> -> !cir.ptr<!u32i>
unsigned load_non_bitfield(S& s) {
  return s.f;
}

// just create a usage of T type
// CHECK: cir.func {{.*}} @_Z17load_one_bitfield
int load_one_bitfield(T& t) {
  return t.a;
}
