// RUN: %clang_cc1 %s -o - -O0 -emit-llvm                                     \
// RUN:            -triple spir64-unknown-unknown                             \
// RUN:            -aux-triple x86_64-unknown-linux-gnu                       \
// RUN:            -fsycl-is-device                                           \
// RUN:            -finclude-default-header                                   \
// RUN:            -debug-info-kind=limited -gno-column-info                  \
// RUN:   | FileCheck %s
//
// In spir functions, validate the llvm.dbg.declare intrinsics created for
// parameters and locals refer to the stack allocation in the alloca address
// space.
//

#define KERNEL __attribute__((sycl_kernel))

template <typename KernelName, typename KernelType>
KERNEL void parallel_for(const KernelType &KernelFunc) {
  KernelFunc();
}

void my_kernel(int my_param) {
  int my_local = 0;
  my_local = my_param;
}

int my_host() {
  parallel_for<class K>([=]() { my_kernel(42); });
  return 0;
}

// CHECK:      define {{.*}}spir_func void @_Z9my_kerneli(
// CHECK-SAME    i32 %my_param
// CHECK-SAME:   !dbg [[MY_KERNEL:![0-9]+]]
// CHECK-SAME: {
// CHECK:        %my_param.addr = alloca i32, align 4
// CHECK:        %my_local = alloca i32, align 4
// CHECK:        call void @llvm.dbg.declare(
// CHECK-SAME:     metadata ptr %my_param.addr,
// CHECK-SAME:     metadata [[MY_PARAM:![0-9]+]],
// CHECK-SAME:     metadata !DIExpression(DW_OP_constu, 4, DW_OP_swap, DW_OP_xderef)
// CHECK-SAME:     )
// CHECK:        call void @llvm.dbg.declare(
// CHECK-SAME:     metadata ptr %my_local,
// CHECK-SAME:     metadata [[MY_LOCAL:![0-9]+]],
// CHECK-SAME:     metadata !DIExpression(DW_OP_constu, 4, DW_OP_swap, DW_OP_xderef)
// CHECK-SAME:     )
// CHECK:      }

// CHECK:      [[MY_KERNEL]] = distinct !DISubprogram(
// CHECK-SAME:   name: "my_kernel"
// CHECK-SAME:   )
// CHECK:      [[MY_PARAM]] = !DILocalVariable(
// CHECK-SAME:   name: "my_param"
// CHECK-SAME:   arg: 1
// CHECK-SAME:   scope: [[MY_KERNEL]]
// CHECK-SAME:   )
// CHECK:      [[MY_LOCAL]] = !DILocalVariable(
// CHECK-SAME:   name: "my_local"
// CHECK-SAME:   scope: [[MY_KERNEL]]
// CHECK-SAME:   )
