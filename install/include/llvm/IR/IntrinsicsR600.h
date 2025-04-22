/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Intrinsic Function Source Fragment                                         *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_IR_INTRINSIC_R600_ENUMS_H
#define LLVM_IR_INTRINSIC_R600_ENUMS_H
namespace llvm::Intrinsic {
enum R600Intrinsics : unsigned {
// Enum values for intrinsics.
    r600_cube = 10841,                                 // llvm.r600.cube
    r600_ddx,                                  // llvm.r600.ddx
    r600_ddy,                                  // llvm.r600.ddy
    r600_dot4,                                 // llvm.r600.dot4
    r600_group_barrier,                        // llvm.r600.group.barrier
    r600_implicitarg_ptr,                      // llvm.r600.implicitarg.ptr
    r600_kill,                                 // llvm.r600.kill
    r600_rat_store_typed,                      // llvm.r600.rat.store.typed
    r600_read_global_size_x,                   // llvm.r600.read.global.size.x
    r600_read_global_size_y,                   // llvm.r600.read.global.size.y
    r600_read_global_size_z,                   // llvm.r600.read.global.size.z
    r600_read_local_size_x,                    // llvm.r600.read.local.size.x
    r600_read_local_size_y,                    // llvm.r600.read.local.size.y
    r600_read_local_size_z,                    // llvm.r600.read.local.size.z
    r600_read_ngroups_x,                       // llvm.r600.read.ngroups.x
    r600_read_ngroups_y,                       // llvm.r600.read.ngroups.y
    r600_read_ngroups_z,                       // llvm.r600.read.ngroups.z
    r600_read_tgid_x,                          // llvm.r600.read.tgid.x
    r600_read_tgid_y,                          // llvm.r600.read.tgid.y
    r600_read_tgid_z,                          // llvm.r600.read.tgid.z
    r600_read_tidig_x,                         // llvm.r600.read.tidig.x
    r600_read_tidig_y,                         // llvm.r600.read.tidig.y
    r600_read_tidig_z,                         // llvm.r600.read.tidig.z
    r600_recipsqrt_clamped,                    // llvm.r600.recipsqrt.clamped
    r600_recipsqrt_ieee,                       // llvm.r600.recipsqrt.ieee
    r600_store_stream_output,                  // llvm.r600.store.stream.output
    r600_store_swizzle,                        // llvm.r600.store.swizzle
    r600_tex,                                  // llvm.r600.tex
    r600_texc,                                 // llvm.r600.texc
    r600_txb,                                  // llvm.r600.txb
    r600_txbc,                                 // llvm.r600.txbc
    r600_txf,                                  // llvm.r600.txf
    r600_txl,                                  // llvm.r600.txl
    r600_txlc,                                 // llvm.r600.txlc
    r600_txq,                                  // llvm.r600.txq
}; // enum
} // namespace llvm::Intrinsic
#endif

