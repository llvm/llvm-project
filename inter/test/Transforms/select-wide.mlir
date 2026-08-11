// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s --check-prefix=MACHINE
// RUN: inter-opt %s --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' | inter-translate --xemachine-to-ged -o %t
// RUN: inter-ged-dump %t | FileCheck %s --check-prefix=GED

module {
  func.func @signed_offset(%base: !llvm.ptr<1>, %index: i32) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, offset = 24, size = 8>,
        #xemachine.kernel_arg<kind = by_value, offset = 32, size = 4>
      ]} {
    %id = xw.global_id 0 : i32
    %zero = llvm.mlir.constant(0 : i32) : i32
    %negative = llvm.icmp "slt" %index, %zero : i32
    %wide = xw.wide_extend %index signed : i32
    %address = xw.ptradd %base, %wide : !llvm.ptr<1>, i64
    %root = xw.token
    %stored = scf.if %negative -> !xemachine.mem.token {
      %then = xw.store %address, %zero dep %root
          : !llvm.ptr<1>, i32 -> !xemachine.mem.token
      scf.yield %then : !xemachine.mem.token
    } else {
      %else = xw.store %address, %id dep %root
          : !llvm.ptr<1>, i32 -> !xemachine.mem.token
      scf.yield %else : !xemachine.mem.token
    }
    %after = xw.barrier dep %stored : !xemachine.mem.token
    return
  }
}

// MACHINE: xemachine.cmp
// MACHINE-SAME: signed
// MACHINE: xemachine.shl
// MACHINE-SAME: signedSource
// MACHINE-SAME: src0Type = i32
// MACHINE-SAME: : ({{.*}}, !xemachine.imm, i64)

// GED: opcode=cmp {{.*}}src0={{.*}}:d<{{.*}}src1=imm0x0:d
// GED: opcode=shl exec=16 {{.*}}src0={{.*}}:d<
// GED: opcode=shl exec=16 {{.*}}src0={{.*}}:d<
