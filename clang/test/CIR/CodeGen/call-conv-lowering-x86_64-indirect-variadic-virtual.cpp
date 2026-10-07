// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct Big { char c[64]; };
struct Base {
  virtual void log(const char *fmt, ...) {}
};

// A struct too large for registers goes byval through the ellipsis of a
// virtual call and stays out of the type of the function pointer loaded from
// the vtable, as for any indirect call.
void call_it(Base *b, Big g) { b->log("%s", g); }

// CIR-LABEL: cir.func {{.*}}@_Z7call_itP4Base3Big(%arg0: !cir.ptr<!rec_Base> {llvm.noundef} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noundef} loc({{[^)]+}}))
// CIR:   %[[SLOT:.*]] = cir.vtable.get_virtual_fn_addr %{{.+}}[0] : !cir.vptr -> !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>, !cir.ptr<!s8i>, ...)>>>
// CIR:   %[[FN:.*]] = cir.load align(8) %[[SLOT]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>, !cir.ptr<!s8i>, ...)>>>, !cir.ptr<!cir.func<(!cir.ptr<!rec_Base>, !cir.ptr<!s8i>, ...)>>
// CIR-NOT: cir.cast bitcast {{.*}}!cir.func
// CIR:   cir.call %[[FN]](%{{.+}}, %{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Base>, !cir.ptr<!s8i>, ...)>>, !cir.ptr<!rec_Base> {llvm.align = 8 : i64, llvm.dereferenceable = 8 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!s8i> {llvm.noundef}, !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noundef}) -> ()

// LLVM-LABEL: define dso_local void @_Z7call_itP4Base3Big(ptr noundef %{{[^,)]+}}, ptr noundef byval(%struct.Big) align 8 %{{[^,)]+}})
// LLVM: call void (ptr, ptr, ...) %{{[^,)]+}}(ptr noundef nonnull align 8 dereferenceable(8) %{{[^,)]+}}, ptr noundef @.str, ptr noundef byval(%struct.Big) align 8 %{{[^,)]+}})
