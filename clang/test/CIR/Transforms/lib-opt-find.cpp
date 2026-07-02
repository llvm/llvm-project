// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir
// RUN: %clang_cc1 -std=c++17 -triple i686-unknown-linux-gnu -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.ilp32.cir
// RUN: FileCheck %s --input-file=%t.ilp32.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnux32 -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.x32.cir
// RUN: FileCheck %s --input-file=%t.x32.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr
// RUN: %clang_cc1 -std=c++17 -triple aarch64-unknown-linux-gnu_ilp32 -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.a64ilp32.cir
// RUN: FileCheck %s --input-file=%t.a64ilp32.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr
// RUN: %clang_cc1 -std=c++17 -triple spirv-unknown-vulkan-compute -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.spirv.cir
// RUN: FileCheck %s --input-file=%t.spirv.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr
// RUN: %clang_cc1 -std=c++17 -triple amdgcn-amd-amdhsa -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.amdgcn.cir
// RUN: FileCheck %s --input-file=%t.amdgcn.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr
// RUN: %clang_cc1 -std=c++17 -triple nvptx64-nvidia-cuda -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.nvptx.cir
// RUN: FileCheck %s --input-file=%t.nvptx.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr
// RUN: %clang_cc1 -std=c++17 -triple bpfel -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.bpf.cir
// RUN: FileCheck %s --input-file=%t.bpf.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -ffreestanding -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.free.cir
// RUN: FileCheck %s --input-file=%t.free.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fno-builtin-memchr -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.nomemchr.cir
// RUN: FileCheck %s --input-file=%t.nomemchr.cir --check-prefix=NOXFORM --implicit-check-not=cir.libc.memchr
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fno-builtin-strlen -fclangir -O1 -clangir-enable-idiom-recognizer -clangir-lib-opt -emit-cir %s -o %t.nostrlen.cir
// RUN: FileCheck %s --input-file=%t.nostrlen.cir
// NOXFORM: cir.call @_ZSt4findIPhhET_S1_S1_RKT0_
// NOXFORM: cir.call @_ZSt4findIPccET_S1_S1_RKT0_
// NOXFORM: cir.call @_ZSt4findIPaaET_S1_S1_RKT0_

namespace std {
template <class Iter, class T> Iter find(Iter, Iter, const T &);
}

unsigned char *test_byte_find(unsigned char *first, unsigned char *last,
                              const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z14test_byte_findPhS_RKh
// CHECK: %[[FIRST:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
// CHECK: %[[LAST:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
// CHECK: %[[VALUEPTR:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
// CHECK-NOT: cir.call
// CHECK: %[[SRC:.*]] = cir.cast bitcast %[[FIRST]] : !cir.ptr<!u8i> -> !cir.ptr<!void>
// CHECK: %[[BYTE:.*]] = cir.load %[[VALUEPTR]] : !cir.ptr<!u8i>, !u8i
// CHECK: %[[PATTERN:.*]] = cir.cast integral %[[BYTE]] : !u8i -> !s32i
// CHECK: %[[LEN:.*]] = cir.ptr_diff %[[LAST]], %[[FIRST]] : !cir.ptr<!u8i> -> !u64i
// CHECK: %[[PTR:.*]] = cir.libc.memchr(%[[SRC]], %[[PATTERN]], %[[LEN]])
// CHECK: %[[RES:.*]] = cir.cast bitcast %[[PTR]] : !cir.ptr<!void> -> !cir.ptr<!u8i>
// CHECK: %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!u8i>
// CHECK: %[[MISS:.*]] = cir.cmp eq %[[RES]], %[[NULL]]
// CHECK: cir.select if %[[MISS]] then %[[LAST]] else %[[RES]]

char *test_char_find(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z14test_char_findPcS_RKc
// CHECK-NOT: cir.call
// CHECK: cir.cast integral %{{.*}} : !s8i -> !s32i
// CHECK: %[[CLEN:.*]] = cir.ptr_diff %{{.*}}, %{{.*}} : !cir.ptr<!s8i> -> !u64i
// CHECK: %[[CPTR:.*]] = cir.libc.memchr(%{{.*}}, %{{.*}}, %[[CLEN]])
// CHECK: %[[CRES:.*]] = cir.cast bitcast %[[CPTR]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
// CHECK: %[[CMISS:.*]] = cir.cmp eq %[[CRES]], %{{.*}}
// CHECK: cir.select if %[[CMISS]] then %{{.*}} else %[[CRES]]

signed char *test_signed_char_find(signed char *first, signed char *last,
                                   const signed char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z21test_signed_char_find
// CHECK-NOT: cir.call
// CHECK: cir.cast integral %{{.*}} : !s8i -> !s32i
// CHECK: %[[SLEN:.*]] = cir.ptr_diff %{{.*}}, %{{.*}} : !cir.ptr<!s8i> -> !u64i
// CHECK: %[[SPTR:.*]] = cir.libc.memchr(%{{.*}}, %{{.*}}, %[[SLEN]])
// CHECK: %[[SRES:.*]] = cir.cast bitcast %[[SPTR]] : !cir.ptr<!void> -> !cir.ptr<!s8i>
// CHECK: %[[SMISS:.*]] = cir.cmp eq %[[SRES]], %{{.*}}
// CHECK: cir.select if %[[SMISS]] then %{{.*}} else %[[SRES]]

int *test_int_find(int *first, int *last, const int &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z13test_int_findPiS_RKi
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

char *test_sign_mismatch(char *first, char *last, const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z18test_sign_mismatchPcS_RKh
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

char *test_char_flavors(char *first, char *last, const signed char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z17test_char_flavorsPcS_RKa
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

volatile unsigned char *test_volatile_find(volatile unsigned char *first,
                                           volatile unsigned char *last,
                                           const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z18test_volatile_findPVhS0_RKh
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

unsigned char *test_volatile_value(unsigned char *first, unsigned char *last,
                                   const volatile unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z19test_volatile_valuePhS_RVKh
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

__attribute__((no_builtin("memchr")))
unsigned char *test_fn_no_builtin(unsigned char *first, unsigned char *last,
                                  const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z18test_fn_no_builtinPhS_RKh
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

enum class Tri : unsigned char { A, B, C };
bool operator==(Tri, Tri);
Tri *test_enum_find(Tri *first, Tri *last, const Tri &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z14test_enum_findP3TriS0_RKS_
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

typedef _Atomic(unsigned char) AByte;
AByte *test_atomic_find(AByte *first, AByte *last,
                        const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z16test_atomic_find
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

struct Byte {
  unsigned char b;
  bool operator==(const Byte &) const;
};
Byte *test_record_find(Byte *first, Byte *last, const Byte &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z16test_record_findP4ByteS0_RKS_
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find

typedef __attribute__((address_space(1))) unsigned char ASByte;
ASByte *test_addr_space_find(ASByte *first, ASByte *last,
                             const unsigned char &value) {
  return std::find(first, last, value);
}
// CHECK-LABEL: @_Z20test_addr_space_findPU3AS1hS0_RKh
// CHECK: cir.call @_ZSt4find{{.*}}(
// CHECK-NOT: cir.libc.memchr
// CHECK-NOT: cir.std.find
