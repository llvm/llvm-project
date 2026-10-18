// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef bool b4 __attribute__((ext_vector_type(4)));
typedef bool b17 __attribute__((ext_vector_type(17)));
typedef bool b24 __attribute__((ext_vector_type(24)));

b4 rb4(b4 v) { return v; }

// CIR: cir.func {{.*}} @_Z3rb4Dv4_b(%arg0: !u8i {llvm.noundef} loc({{[^)]+}})) -> (!u8i {llvm.noundef})
// LLVM: define dso_local noundef i8 @_Z3rb4Dv4_b(i8 noundef %{{[^,)]+}})

// The 17-bit storage integer is not a whole number of bytes, so the return is
// not noundef.
b17 rb17(b17 v) { return v; }

// CIR: cir.func {{.*}} @_Z4rb17Dv17_b(%arg0: !u32i loc({{[^)]+}})) -> !u32i
// LLVM: define dso_local i32 @_Z4rb17Dv17_b(i32 %{{[^,)]+}})

// The i32 is wider than the 24 bits of storage, so the return is not noundef.
b24 rb24(b24 v) { return v; }

// CIR: cir.func {{.*}} @_Z4rb24Dv24_b(%arg0: !u32i loc({{[^)]+}})) -> !u32i
// LLVM: define dso_local i32 @_Z4rb24Dv24_b(i32 %{{[^,)]+}})

b4 call_rb4(b4 v) { return rb4(v); }

// CIR: cir.call @_Z3rb4Dv4_b(%{{[^)]+}}) : (!u8i {llvm.noundef}) -> (!u8i {llvm.noundef})
// LLVM: call noundef i8 @_Z3rb4Dv4_b(i8 noundef %{{[^,)]+}})

b24 call_rb24(b24 v) { return rb24(v); }

// CIR: cir.call @_Z4rb24Dv24_b(%{{[^)]+}}) : (!u32i) -> !u32i
// LLVM: call i32 @_Z4rb24Dv24_b(i32 %{{[^,)]+}})
