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

void m() {
  __long l;
}

// CHECK: !ty_22anon22 = !cir.struct<struct "anon" {!u32i} #cir.record.decl.ast>
// CHECK: !ty_22__long22 = !cir.struct<struct "__long" {!ty_22anon22, !u32i, !cir.ptr<!u32i>}>
