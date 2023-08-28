// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I%S/../Inputs -clangir-disable-emit-cxx-default -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

#include "std-cxx.h"

typedef enum enumy {
  Unknown = 0,
  Some = 1000024002,
} enumy;

typedef struct triple {
  enumy type;
  void* __attribute__((__may_alias__)) next;
  unsigned image;
} triple;

void init(unsigned numImages) {
  std::vector<triple> images(numImages);
  for (auto& image : images) {
    image = {Some};
  }
}

// CHECK: !ty_22struct2Etriple22 = !cir.struct<"struct.triple" {!u32i, !cir.ptr<!void>, !u32i}>
// CHECK: !ty_22class2Estd3A3Avector22 = !cir.struct<"class.std::vector" {!cir.ptr<!ty_22struct2Etriple22>, !cir.ptr<!ty_22struct2Etriple22>, !cir.ptr<!ty_22struct2Etriple22>}>
// CHECK: !ty_22struct2E__vector_iterator22 = !cir.struct<"struct.__vector_iterator" {!cir.ptr<!ty_22struct2Etriple22>}>

// CHECK: cir.func @_Z4initj(%arg0: !u32i
// CHECK:   %0 = cir.alloca !u32i, cir.ptr <!u32i>, ["numImages", init] {alignment = 4 : i64}
// CHECK:   %1 = cir.alloca !ty_22class2Estd3A3Avector22, cir.ptr <!ty_22class2Estd3A3Avector22>, ["images", init] {alignment = 8 : i64}
// CHECK:   cir.store %arg0, %0 : !u32i, cir.ptr <!u32i>
// CHECK:   %2 = cir.load %0 : cir.ptr <!u32i>, !u32i
// CHECK:   %3 = cir.cast(integral, %2 : !u32i), !u64i
// CHECK:   cir.call @_ZNSt6vectorI6tripleEC1Em(%1, %3) : (!cir.ptr<!ty_22class2Estd3A3Avector22>, !u64i) -> ()
// CHECK:   cir.scope {
// CHECK:     %4 = cir.alloca !cir.ptr<!ty_22class2Estd3A3Avector22>, cir.ptr <!cir.ptr<!ty_22class2Estd3A3Avector22>>, ["__range1", init] {alignment = 8 : i64}
// CHECK:     %5 = cir.alloca !ty_22struct2E__vector_iterator22, cir.ptr <!ty_22struct2E__vector_iterator22>, ["__begin1", init] {alignment = 8 : i64}
// CHECK:     %6 = cir.alloca !ty_22struct2E__vector_iterator22, cir.ptr <!ty_22struct2E__vector_iterator22>, ["__end1", init] {alignment = 8 : i64}
// CHECK:     %7 = cir.alloca !cir.ptr<!ty_22struct2Etriple22>, cir.ptr <!cir.ptr<!ty_22struct2Etriple22>>, ["image", init] {alignment = 8 : i64}
// CHECK:     cir.store %1, %4 : !cir.ptr<!ty_22class2Estd3A3Avector22>, cir.ptr <!cir.ptr<!ty_22class2Estd3A3Avector22>>
// CHECK:     %8 = cir.load %4 : cir.ptr <!cir.ptr<!ty_22class2Estd3A3Avector22>>, !cir.ptr<!ty_22class2Estd3A3Avector22>
// CHECK:     %9 = cir.call @_ZNSt6vectorI6tripleE5beginEv(%8) : (!cir.ptr<!ty_22class2Estd3A3Avector22>) -> !ty_22struct2E__vector_iterator22
// CHECK:     cir.store %9, %5 : !ty_22struct2E__vector_iterator22, cir.ptr <!ty_22struct2E__vector_iterator22>
// CHECK:     %10 = cir.load %4 : cir.ptr <!cir.ptr<!ty_22class2Estd3A3Avector22>>, !cir.ptr<!ty_22class2Estd3A3Avector22>
// CHECK:     %11 = cir.call @_ZNSt6vectorI6tripleE3endEv(%10) : (!cir.ptr<!ty_22class2Estd3A3Avector22>) -> !ty_22struct2E__vector_iterator22
// CHECK:     cir.store %11, %6 : !ty_22struct2E__vector_iterator22, cir.ptr <!ty_22struct2E__vector_iterator22>
// CHECK:     cir.loop for(cond : {
// CHECK:       %12 = cir.call @_ZNK17__vector_iteratorI6triplePS0_RS0_EneERKS3_(%5, %6) : (!cir.ptr<!ty_22struct2E__vector_iterator22>, !cir.ptr<!ty_22struct2E__vector_iterator22>) -> !cir.bool
// CHECK:       cir.brcond %12 ^bb1, ^bb2
// CHECK:     ^bb1:  // pred: ^bb0
// CHECK:       cir.yield continue
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       cir.yield
// CHECK:     }, step : {
// CHECK:       %12 = cir.call @_ZN17__vector_iteratorI6triplePS0_RS0_EppEv(%5) : (!cir.ptr<!ty_22struct2E__vector_iterator22>) -> !cir.ptr<!ty_22struct2E__vector_iterator22>
// CHECK:       cir.yield
// CHECK:     }) {
// CHECK:       %12 = cir.call @_ZNK17__vector_iteratorI6triplePS0_RS0_EdeEv(%5) : (!cir.ptr<!ty_22struct2E__vector_iterator22>) -> !cir.ptr<!ty_22struct2Etriple22>
// CHECK:       cir.store %12, %7 : !cir.ptr<!ty_22struct2Etriple22>, cir.ptr <!cir.ptr<!ty_22struct2Etriple22>>
// CHECK:       cir.scope {
// CHECK:         %13 = cir.alloca !ty_22struct2Etriple22, cir.ptr <!ty_22struct2Etriple22>, ["ref.tmp0"] {alignment = 8 : i64}
// CHECK:         %14 = cir.const(#cir.zero : !ty_22struct2Etriple22) : !ty_22struct2Etriple22
// CHECK:         cir.store %14, %13 : !ty_22struct2Etriple22, cir.ptr <!ty_22struct2Etriple22>
// CHECK:         %15 = "cir.struct_element_addr"(%13) <{member_index = 0 : index, member_name = "type"}> : (!cir.ptr<!ty_22struct2Etriple22>) -> !cir.ptr<!u32i>
// CHECK:         %16 = cir.const(#cir.int<1000024002> : !u32i) : !u32i
// CHECK:         cir.store %16, %15 : !u32i, cir.ptr <!u32i>
// CHECK:         %17 = "cir.struct_element_addr"(%13) <{member_index = 1 : index, member_name = "next"}> : (!cir.ptr<!ty_22struct2Etriple22>) -> !cir.ptr<!cir.ptr<!void>>
// CHECK:         %18 = "cir.struct_element_addr"(%13) <{member_index = 2 : index, member_name = "image"}> : (!cir.ptr<!ty_22struct2Etriple22>) -> !cir.ptr<!u32i>
// CHECK:         %19 = cir.load %7 : cir.ptr <!cir.ptr<!ty_22struct2Etriple22>>, !cir.ptr<!ty_22struct2Etriple22>
// CHECK:         %20 = cir.call @_ZN6tripleaSEOS_(%19, %13) : (!cir.ptr<!ty_22struct2Etriple22>, !cir.ptr<!ty_22struct2Etriple22>) -> !cir.ptr<!ty_22struct2Etriple22>
// CHECK:       }
// CHECK:       cir.yield
// CHECK:     }
// CHECK:   }
// CHECK:   cir.return
