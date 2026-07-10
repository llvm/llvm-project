// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

void noargs(...) {
  __builtin_va_list list;
  __builtin_va_start(list, 0);
  __builtin_c23_va_start(list);
  __builtin_va_end(list);
}

// CIR-LABEL: cir.func {{.*}} @noargs(
// CIR:   %[[VAAREA:.+]] = cir.alloca "list" {{.*}} : !cir.ptr<!cir.array<!rec___va_list_tag x 1>>
// CIR:   %[[VA_PTR0:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]]
// CIR-NEXT:   cir.va_start %[[VA_PTR0]] : !cir.ptr<!rec___va_list_tag>
// CIR:   %[[VA_PTR1:.+]] = cir.cast array_to_ptrdecay %[[VAAREA]]
// CIR-NEXT:   cir.va_start %[[VA_PTR1]] : !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %{{.+}} : !cir.ptr<!rec___va_list_tag>

// LLVM-LABEL: define {{.*}}void @noargs(...)
// LLVM:   %[[VAAREA:.+]] = alloca [1 x %struct.__va_list_tag]
// LLVM:   call void @llvm.va_start.p0(ptr %{{.+}})
// LLVM:   call void @llvm.va_start.p0(ptr %{{.+}})
// LLVM:   call void @llvm.va_end.p0(ptr %{{.+}})

void with_param(int count, ...) {
  __builtin_va_list list;
  __builtin_c23_va_start(list, count);
  __builtin_va_end(list);
}

// CIR-LABEL: cir.func {{.*}} @with_param(
// CIR:   cir.va_start %{{.+}} : !cir.ptr<!rec___va_list_tag>
// CIR:   cir.va_end %{{.+}} : !cir.ptr<!rec___va_list_tag>

// LLVM-LABEL: define {{.*}}void @with_param(i32 noundef %{{.+}}, ...)
// LLVM:   call void @llvm.va_start.p0(ptr %{{.+}})
// LLVM:   call void @llvm.va_end.p0(ptr %{{.+}})
