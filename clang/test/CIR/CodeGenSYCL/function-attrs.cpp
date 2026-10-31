// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes -triple spir64 -fexceptions -fno-ident -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes -triple spir64 -fexceptions -fno-ident -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -fsycl-is-device -disable-llvm-passes -triple spir64 -fexceptions -fno-ident -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

int foo();

// CIR:      cir.func {{.*}}@_Z3barv() cc(spir_function) attributes {convergent, mustprogress, norecurse, nothrow, nounwind, optnone} {
// CIR-NEXT:   %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT:   %[[A_ASCAST:.*]] = cir.cast address_space %[[A]] : !cir.ptr<!s32i> -> !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT:   %[[CALL:.*]] = cir.call @_Z3foov() cc(spir_function) nothrow nounwind {convergent} : () -> (!s32i {llvm.noundef})
// CIR-NEXT:   cir.store align(4) %[[CALL]], %[[A_ASCAST]] : !s32i, !cir.ptr<!s32i, target_address_space(4)>
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// LLVM:      define dso_local spir_func void @_Z3barv() #[[ATTR0:[0-9]+]] {
// OGCG-NEXT: entry:
// LLVM-NEXT:   %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT:   %[[A_ASCAST:.*]] = addrspacecast ptr %[[A]] to ptr addrspace(4)
// LLVM-NEXT:   %[[CALL:.*]] = call spir_func noundef i32 @_Z3foov() #[[ATTR1:[0-9]+]]
// LLVM-NEXT:   store i32 %[[CALL]], ptr addrspace(4) %[[A_ASCAST]], align 4
// LLVM-NEXT:   ret void
// LLVM-NEXT: }
void bar() {
  int a = foo();
}

// CIR:      cir.func {{.*}}@_Z3foov() -> (!s32i {llvm.noundef}) cc(spir_function) attributes {convergent, mustprogress, norecurse, nothrow, nounwind, optnone} {
// CIR-NEXT:   %[[RETVAL:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:   cir.store %[[ONE]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:   cir.return %[[RET]] : !s32i
// CIR-NEXT: }

// ClangIR returns through a __retval slot, while classic CodeGen folds the
// single return into the entry block.
// LLVM:          define dso_local spir_func noundef i32 @_Z3foov() #[[ATTR0]] {
// LLVM-CIR-NEXT:   %[[RETVAL:.*]] = alloca i32, align 4
// LLVM-CIR-NEXT:   store i32 1, ptr %[[RETVAL]], align 4
// LLVM-CIR-NEXT:   %[[RET:.*]] = load i32, ptr %[[RETVAL]], align 4
// LLVM-CIR-NEXT:   ret i32 %[[RET]]
// OGCG-NEXT:     entry:
// OGCG-NEXT:       ret i32 1
// LLVM-NEXT:     }
int foo() {
  return 1;
}

template <typename Name, typename... Ts>
void sycl_kernel_launch(Ts...) {}

template <typename Name, typename Func>
[[clang::sycl_kernel_entry_point(Name)]] void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([] { bar(); });
  return 0;
}

// ClangIR does not yet emit the "no-trapping-math" and
// "stack-protector-buffer-size" default function attributes.
// LLVM-CIR: attributes #[[ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone }
// OGCG: attributes #[[ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// LLVM: attributes #[[ATTR1]] = { convergent nounwind }
