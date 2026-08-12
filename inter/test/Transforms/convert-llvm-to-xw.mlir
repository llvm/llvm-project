// RUN: inter-opt %s --inter-import-llvm --lift-cf-to-scf \
// RUN:   --inter-verify-structured --inter-convert-llvm-to-xw | FileCheck %s

module {
  llvm.func spir_kernelcc @vector_add(%out: !llvm.ptr<1>,
                                      %in: !llvm.ptr<1>) {
    %axis = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%axis) : (i32) -> i64
    %input_ptr = llvm.getelementptr %in[%gid]
        : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %output_ptr = llvm.getelementptr %out[%gid]
        : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %value = llvm.load %input_ptr : !llvm.ptr<1> -> i32
    %sum = llvm.add %value, %one : i32
    llvm.store %sum, %output_ptr : i32, !llvm.ptr<1>
    llvm.return
  }
  llvm.func spir_funccc @_Z13get_global_idj(i32) -> i64

  llvm.mlir.global internal @scratch() {addr_space = 3 : i32} : i32

  llvm.func spir_kernelcc @spaces(%p0: !llvm.ptr, %p1: !llvm.ptr<1>,
                                  %p2: !llvm.ptr<2>, %p3: !llvm.ptr<3>,
                                  %p4: !llvm.ptr<4>, %condition: i1) {
    %local = llvm.mlir.addressof @scratch : !llvm.ptr<3>
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %sum = llvm.add %zero, %one : i32
    %wide = llvm.sext %sum : i32 to i64
    %cmp = llvm.icmp "eq" %sum, %one : i32
    %selected = llvm.select %cmp, %sum, %one : i1, i32
    llvm.cond_br %condition, ^then, ^else
  ^then:
    llvm.br ^merge(%selected : i32)
  ^else:
    llvm.br ^merge(%zero : i32)
  ^merge(%result: i32):
    llvm.store %result, %local : i32, !llvm.ptr<3>
    llvm.return
  }
}

// CHECK-LABEL: func.func @vector_add(%{{.*}}: !xw.ptr<#xw.global>
// CHECK: xw.global_id 0
// CHECK: xw.ptradd
// CHECK: xw.load
// CHECK: xw.binary addi
// CHECK: xw.store
// CHECK-NOT: llvm.
// CHECK-NOT: {{(^|[^s])cf\.}}

// CHECK-LABEL: func.func @spaces(
// CHECK-SAME: !xw.ptr<#xw.private>
// CHECK-SAME: !xw.ptr<#xw.global>
// CHECK-SAME: !xw.ptr<#xw.constant>
// CHECK-SAME: !xw.ptr<#xw.local>
// CHECK-SAME: !xw.ptr<#xw.generic>
// CHECK: xw.local_memory_base
// CHECK: xw.cast intconvert
// CHECK: xw.cmpi eq
// CHECK: xw.select
// CHECK: scf.if
// CHECK-NOT: llvm.
// CHECK-NOT: {{(^|[^s])cf\.}}
