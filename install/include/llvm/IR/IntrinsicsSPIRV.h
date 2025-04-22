/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Intrinsic Function Source Fragment                                         *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_IR_INTRINSIC_SPV_ENUMS_H
#define LLVM_IR_INTRINSIC_SPV_ENUMS_H
namespace llvm::Intrinsic {
enum SPVIntrinsics : unsigned {
// Enum values for intrinsics.
    spv_all = 11847,                                   // llvm.spv.all
    spv_alloca,                                // llvm.spv.alloca
    spv_alloca_array,                          // llvm.spv.alloca.array
    spv_any,                                   // llvm.spv.any
    spv_assign_aliasing_decoration,            // llvm.spv.assign.aliasing.decoration
    spv_assign_decoration,                     // llvm.spv.assign.decoration
    spv_assign_fpmaxerror_decoration,          // llvm.spv.assign.fpmaxerror.decoration
    spv_assign_name,                           // llvm.spv.assign.name
    spv_assign_ptr_type,                       // llvm.spv.assign.ptr.type
    spv_assign_type,                           // llvm.spv.assign.type
    spv_assume,                                // llvm.spv.assume
    spv_bitcast,                               // llvm.spv.bitcast
    spv_cmpxchg,                               // llvm.spv.cmpxchg
    spv_const_composite,                       // llvm.spv.const.composite
    spv_cross,                                 // llvm.spv.cross
    spv_degrees,                               // llvm.spv.degrees
    spv_discard,                               // llvm.spv.discard
    spv_distance,                              // llvm.spv.distance
    spv_dot4add_i8packed,                      // llvm.spv.dot4add.i8packed
    spv_dot4add_u8packed,                      // llvm.spv.dot4add.u8packed
    spv_expect,                                // llvm.spv.expect
    spv_extractelt,                            // llvm.spv.extractelt
    spv_extractv,                              // llvm.spv.extractv
    spv_fdot,                                  // llvm.spv.fdot
    spv_firstbitlow,                           // llvm.spv.firstbitlow
    spv_firstbitshigh,                         // llvm.spv.firstbitshigh
    spv_firstbituhigh,                         // llvm.spv.firstbituhigh
    spv_flattened_thread_id_in_group,          // llvm.spv.flattened.thread.id.in.group
    spv_frac,                                  // llvm.spv.frac
    spv_gep,                                   // llvm.spv.gep
    spv_group_id,                              // llvm.spv.group.id
    spv_group_memory_barrier_with_group_sync,  // llvm.spv.group.memory.barrier.with.group.sync
    spv_init_global,                           // llvm.spv.init.global
    spv_inline_asm,                            // llvm.spv.inline.asm
    spv_insertelt,                             // llvm.spv.insertelt
    spv_insertv,                               // llvm.spv.insertv
    spv_length,                                // llvm.spv.length
    spv_lerp,                                  // llvm.spv.lerp
    spv_lifetime_end,                          // llvm.spv.lifetime.end
    spv_lifetime_start,                        // llvm.spv.lifetime.start
    spv_load,                                  // llvm.spv.load
    spv_loop_merge,                            // llvm.spv.loop.merge
    spv_nclamp,                                // llvm.spv.nclamp
    spv_normalize,                             // llvm.spv.normalize
    spv_ptrcast,                               // llvm.spv.ptrcast
    spv_radians,                               // llvm.spv.radians
    spv_reflect,                               // llvm.spv.reflect
    spv_resource_getpointer,                   // llvm.spv.resource.getpointer
    spv_resource_handlefrombinding,            // llvm.spv.resource.handlefrombinding
    spv_resource_load_typedbuffer,             // llvm.spv.resource.load.typedbuffer
    spv_resource_store_typedbuffer,            // llvm.spv.resource.store.typedbuffer
    spv_resource_updatecounter,                // llvm.spv.resource.updatecounter
    spv_rsqrt,                                 // llvm.spv.rsqrt
    spv_saturate,                              // llvm.spv.saturate
    spv_sclamp,                                // llvm.spv.sclamp
    spv_sdot,                                  // llvm.spv.sdot
    spv_selection_merge,                       // llvm.spv.selection.merge
    spv_sign,                                  // llvm.spv.sign
    spv_smoothstep,                            // llvm.spv.smoothstep
    spv_step,                                  // llvm.spv.step
    spv_store,                                 // llvm.spv.store
    spv_switch,                                // llvm.spv.switch
    spv_thread_id,                             // llvm.spv.thread.id
    spv_thread_id_in_group,                    // llvm.spv.thread.id.in.group
    spv_track_constant,                        // llvm.spv.track.constant
    spv_uclamp,                                // llvm.spv.uclamp
    spv_udot,                                  // llvm.spv.udot
    spv_undef,                                 // llvm.spv.undef
    spv_unreachable,                           // llvm.spv.unreachable
    spv_unref_global,                          // llvm.spv.unref.global
    spv_value_md,                              // llvm.spv.value.md
    spv_wave_active_countbits,                 // llvm.spv.wave.active.countbits
    spv_wave_all,                              // llvm.spv.wave.all
    spv_wave_any,                              // llvm.spv.wave.any
    spv_wave_is_first_lane,                    // llvm.spv.wave.is.first.lane
    spv_wave_readlane,                         // llvm.spv.wave.readlane
    spv_wave_reduce_max,                       // llvm.spv.wave.reduce.max
    spv_wave_reduce_sum,                       // llvm.spv.wave.reduce.sum
    spv_wave_reduce_umax,                      // llvm.spv.wave.reduce.umax
}; // enum
} // namespace llvm::Intrinsic
#endif

