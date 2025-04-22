/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Intrinsic Function Source Fragment                                         *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_IR_INTRINSIC_DX_ENUMS_H
#define LLVM_IR_INTRINSIC_DX_ENUMS_H
namespace llvm::Intrinsic {
enum DXIntrinsics : unsigned {
// Enum values for intrinsics.
    dx_all = 3925,                                    // llvm.dx.all
    dx_any,                                    // llvm.dx.any
    dx_asdouble,                               // llvm.dx.asdouble
    dx_cross,                                  // llvm.dx.cross
    dx_degrees,                                // llvm.dx.degrees
    dx_discard,                                // llvm.dx.discard
    dx_dot2,                                   // llvm.dx.dot2
    dx_dot2add,                                // llvm.dx.dot2add
    dx_dot3,                                   // llvm.dx.dot3
    dx_dot4,                                   // llvm.dx.dot4
    dx_dot4add_i8packed,                       // llvm.dx.dot4add.i8packed
    dx_dot4add_u8packed,                       // llvm.dx.dot4add.u8packed
    dx_fdot,                                   // llvm.dx.fdot
    dx_firstbitlow,                            // llvm.dx.firstbitlow
    dx_firstbitshigh,                          // llvm.dx.firstbitshigh
    dx_firstbituhigh,                          // llvm.dx.firstbituhigh
    dx_flattened_thread_id_in_group,           // llvm.dx.flattened.thread.id.in.group
    dx_frac,                                   // llvm.dx.frac
    dx_group_id,                               // llvm.dx.group.id
    dx_group_memory_barrier_with_group_sync,   // llvm.dx.group.memory.barrier.with.group.sync
    dx_imad,                                   // llvm.dx.imad
    dx_isinf,                                  // llvm.dx.isinf
    dx_lerp,                                   // llvm.dx.lerp
    dx_nclamp,                                 // llvm.dx.nclamp
    dx_normalize,                              // llvm.dx.normalize
    dx_radians,                                // llvm.dx.radians
    dx_resource_casthandle,                    // llvm.dx.resource.casthandle
    dx_resource_getpointer,                    // llvm.dx.resource.getpointer
    dx_resource_handlefrombinding,             // llvm.dx.resource.handlefrombinding
    dx_resource_load_cbufferrow_2,             // llvm.dx.resource.load.cbufferrow.2
    dx_resource_load_cbufferrow_4,             // llvm.dx.resource.load.cbufferrow.4
    dx_resource_load_cbufferrow_8,             // llvm.dx.resource.load.cbufferrow.8
    dx_resource_load_rawbuffer,                // llvm.dx.resource.load.rawbuffer
    dx_resource_load_typedbuffer,              // llvm.dx.resource.load.typedbuffer
    dx_resource_store_rawbuffer,               // llvm.dx.resource.store.rawbuffer
    dx_resource_store_typedbuffer,             // llvm.dx.resource.store.typedbuffer
    dx_resource_updatecounter,                 // llvm.dx.resource.updatecounter
    dx_rsqrt,                                  // llvm.dx.rsqrt
    dx_saturate,                               // llvm.dx.saturate
    dx_sclamp,                                 // llvm.dx.sclamp
    dx_sdot,                                   // llvm.dx.sdot
    dx_sign,                                   // llvm.dx.sign
    dx_splitdouble,                            // llvm.dx.splitdouble
    dx_step,                                   // llvm.dx.step
    dx_thread_id,                              // llvm.dx.thread.id
    dx_thread_id_in_group,                     // llvm.dx.thread.id.in.group
    dx_uclamp,                                 // llvm.dx.uclamp
    dx_udot,                                   // llvm.dx.udot
    dx_umad,                                   // llvm.dx.umad
    dx_wave_active_countbits,                  // llvm.dx.wave.active.countbits
    dx_wave_all,                               // llvm.dx.wave.all
    dx_wave_any,                               // llvm.dx.wave.any
    dx_wave_getlaneindex,                      // llvm.dx.wave.getlaneindex
    dx_wave_is_first_lane,                     // llvm.dx.wave.is.first.lane
    dx_wave_readlane,                          // llvm.dx.wave.readlane
    dx_wave_reduce_max,                        // llvm.dx.wave.reduce.max
    dx_wave_reduce_sum,                        // llvm.dx.wave.reduce.sum
    dx_wave_reduce_umax,                       // llvm.dx.wave.reduce.umax
    dx_wave_reduce_usum,                       // llvm.dx.wave.reduce.usum
}; // enum
} // namespace llvm::Intrinsic
#endif

