// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

[[gnu::pure]] int pure_func(int x);
[[gnu::const]] int const_func(int x);

int test(int x) {
  int y1 = pure_func(x);
  int y2 = const_func(x);
  return y1 + y2;
}

// CIR-LABEL: @_Z4testi
// CIR:   %{{.+}} = cir.call @_Z9pure_funci(%{{.+}}) : (!s32i) -> !s32i side_effect(pure)
// CIR:   %{{.+}} = cir.call @_Z10const_funci(%{{.+}}) : (!s32i) -> !s32i side_effect(const)
// CIR: }

// LLVM-LABEL: @_Z4testi(i32 %0)
// LLVM:   %{{.+}} = call i32 @_Z9pure_funci(i32 %{{.+}}) #[[#meta_pure:]]
// LLVM:   %{{.+}} = call i32 @_Z10const_funci(i32 %{{.+}}) #[[#meta_const:]]
// LLVM: }
// LLVM: attributes #[[#meta_pure]] = { nounwind willreturn memory(read) }
// LLVM: attributes #[[#meta_const]] = { nounwind willreturn memory(none) }
