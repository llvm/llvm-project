// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct __long {
  struct __attribute__((__packed__)) {
      unsigned __is_long_ : 1;
      unsigned __cap_ : sizeof(unsigned) * 8 - 1;
  };
  unsigned __size_;
  unsigned *__data_;
};

void m() {
  struct __long l;
}

typedef struct {
  int a : 4;
  int b : 5;
  int c;
} D;

typedef struct {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e : 15;
  unsigned f; // type other than int above, not a bitfield
} S;

typedef struct {
  int a : 3;  // one bitfield with size < 8
  unsigned b;
} T;

typedef struct {
    char a;
    char b;
    char c;

    // startOffset 24 bits, new storage from here
    int d: 2;
    int e: 2;
    int f: 4;
    int g: 25;
    int h: 3;
    int i: 4;
    int j: 3;
    int k: 8;

    int l: 14; // need to be a part of the new storage
               // because (tail - startOffset) is 65 after 'l' field
} U;

// CHECK: !ty_D = !cir.struct<struct "D" {!u16i, !s32i}>
// CHECK: !ty_G = !cir.struct<struct "G" {!u16i, !s32i} #cir.record.decl.ast>
// CHECK: !ty_T = !cir.struct<struct "T" {!u8i, !u32i} #cir.record.decl.ast>
// CHECK: !ty_anon2E0_ = !cir.struct<struct "anon.0" {!u32i} #cir.record.decl.ast>
// CHECK: #bfi_a = #cir.bitfield_info<name = "a", storage_type = !u8i, size = 3, offset = 0, is_signed = true>
// CHECK: #bfi_e = #cir.bitfield_info<name = "e", storage_type = !u16i, size = 15, offset = 0, is_signed = true>
// CHECK: !ty_S = !cir.struct<struct "S" {!u32i, !cir.array<!u8i x 3>, !u16i, !u32i}>
// CHECK: !ty_U = !cir.struct<struct "U" {!s8i, !s8i, !s8i, !cir.array<!u8i x 9>}>
// CHECK: !ty___long = !cir.struct<struct "__long" {!ty_anon2E0_, !u32i, !cir.ptr<!u32i>}>
// CHECK: !ty_anon_struct = !cir.struct<struct  {!u8i, !u8i, !cir.array<!u8i x 2>, !s32i}>
// CHECK: #bfi_d = #cir.bitfield_info<name = "d", storage_type = !cir.array<!u8i x 3>, size = 2, offset = 17, is_signed = true>

// CHECK: cir.func {{.*@store_field}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !ty_S, !cir.ptr<!ty_S>
// CHECK:   [[TMP1:%.*]] = cir.const #cir.int<3> : !s32i
// CHECK:   [[TMP2:%.*]] = cir.get_member [[TMP0]][2] {name = "e"} : !cir.ptr<!ty_S> -> !cir.ptr<!u16i>
// CHECK:   cir.set_bitfield(#bfi_e, [[TMP2]] : !cir.ptr<!u16i>, [[TMP1]] : !s32i)
void store_field() {
  S s;
  s.e = 3;
}

// CHECK: cir.func {{.*@load_field}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!ty_S>, !cir.ptr<!cir.ptr<!ty_S>>, ["s", init]
// CHECK:   [[TMP1:%.*]] = cir.load [[TMP0]] : !cir.ptr<!cir.ptr<!ty_S>>, !cir.ptr<!ty_S>
// CHECK:   [[TMP2:%.*]] = cir.get_member [[TMP1]][1] {name = "d"} : !cir.ptr<!ty_S> -> !cir.ptr<!cir.array<!u8i x 3>>
// CHECK:   [[TMP3:%.*]] = cir.get_bitfield(#bfi_d, [[TMP2]] : !cir.ptr<!cir.array<!u8i x 3>>) -> !s32i
int load_field(S* s) {
  return s->d;
}

// CHECK: cir.func {{.*@unOp}}
// CHECK:   [[TMP0:%.*]] = cir.get_member {{.*}}[1] {name = "d"} : !cir.ptr<!ty_S> -> !cir.ptr<!cir.array<!u8i x 3>>
// CHECK:   [[TMP1:%.*]] = cir.get_bitfield(#bfi_d, [[TMP0]] : !cir.ptr<!cir.array<!u8i x 3>>) -> !s32i
// CHECK:   [[TMP2:%.*]] = cir.unary(inc, [[TMP1]]) : !s32i, !s32i
// CHECK:   cir.set_bitfield(#bfi_d, [[TMP0]] : !cir.ptr<!cir.array<!u8i x 3>>, [[TMP2]] : !s32i)
void unOp(S* s) {
  s->d++;
}

// CHECK: cir.func {{.*@binOp}}
// CHECK:   [[TMP0:%.*]] = cir.const #cir.int<42> : !s32i
// CHECK:   [[TMP1:%.*]] = cir.get_member {{.*}}[1] {name = "d"} : !cir.ptr<!ty_S> -> !cir.ptr<!cir.array<!u8i x 3>>
// CHECK:   [[TMP2:%.*]] = cir.get_bitfield(#bfi_d, [[TMP1]] : !cir.ptr<!cir.array<!u8i x 3>>) -> !s32i
// CHECK:   [[TMP3:%.*]] = cir.binop(or, [[TMP2]], [[TMP0]]) : !s32i
// CHECK:   cir.set_bitfield(#bfi_d, [[TMP1]] : !cir.ptr<!cir.array<!u8i x 3>>, [[TMP3]] : !s32i)
void binOp(S* s) {
   s->d |= 42;
}


// CHECK: cir.func {{.*@load_non_bitfield}}
// CHECK:   cir.get_member {{%.}}[3] {name = "f"} : !cir.ptr<!ty_S> -> !cir.ptr<!u32i>
unsigned load_non_bitfield(S *s) {
  return s->f;
}

// just create a usage of T type
// CHECK: cir.func {{.*@load_one_bitfield}}
int load_one_bitfield(T* t) {
  return t->a;
}

// CHECK: cir.func {{.*@createU}}
void createU() {
  U u;
}

// for this struct type we create an anon structure with different storage types in initialization
// CHECK: cir.func {{.*@createD}}
// CHECK:   %0 = cir.alloca !ty_D, !cir.ptr<!ty_D>, ["d"] {alignment = 4 : i64}
// CHECK:   %1 = cir.cast(bitcast, %0 : !cir.ptr<!ty_D>), !cir.ptr<!ty_anon_struct>
// CHECK:   %2 = cir.const #cir.const_struct<{#cir.int<33> : !u8i, #cir.int<0> : !u8i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 2>, #cir.int<3> : !s32i}> : !ty_anon_struct
// CHECK:   cir.store %2, %1 : !ty_anon_struct, !cir.ptr<!ty_anon_struct>
void createD() {
  D d = {1,2,3};
}

// check the -1 is stored to the ret value
// LLVM: define dso_local i32 {{@.*get_a.*}}
// LLVM:    %[[V1:.*]] = alloca i32
// LLVM:    store i32 -1, ptr %[[V1]], align 4
// LLVM:    %[[V2:.*]] = load i32, ptr %[[V1]], align 4
// LLVM:    ret i32 %[[V2:.*]]
int get_a(T *t) {
  return (t->a = 7);
}

typedef struct {
  int x : 15;
  int y ;
} G;

// CHECK: cir.global external @g = #cir.const_struct<{#cir.int<133> : !u8i, #cir.int<127> : !u8i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 2>, #cir.int<254> : !s32i}> : !ty_anon_struct
G g = { -123, 254UL};

// CHECK: cir.func {{.*@get_y}}
// CHECK:   %[[V1:.*]] = cir.get_global @g : !cir.ptr<!ty_anon_struct>
// CHECK:   %[[V2:.*]] = cir.cast(bitcast, %[[V1]] : !cir.ptr<!ty_anon_struct>), !cir.ptr<!ty_G>
// CHECK:   %[[V3:.*]] = cir.get_member %[[V2]][1] {name = "y"} : !cir.ptr<!ty_G> -> !cir.ptr<!s32i>
// CHECK:   cir.load %[[V3]] : !cir.ptr<!s32i>, !s32i
int get_y() {
  return g.y;
}
