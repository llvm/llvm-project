// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;

u16 bswap_u16(u16 x) {
  return __builtin_bswap16(x);
}

// CHECK: cir.func @_Z9bswap_u16t
// CHECK:   %{{.+}} = cir.bswap(%{{.+}} : !u16i) : !u16i
// CHECK: }

u32 bswap_u32(u32 x) {
  return __builtin_bswap32(x);
}

// CHECK: cir.func @_Z9bswap_u32j
// CHECK:   %{{.+}} = cir.bswap(%{{.+}} : !u32i) : !u32i
// CHECK: }

u64 bswap_u64(u64 x) {
  return __builtin_bswap64(x);
}

// CHECK: cir.func @_Z9bswap_u64y
// CHECK:   %{{.+}} = cir.bswap(%{{.+}} : !u64i) : !u64i
// CHECK: }
