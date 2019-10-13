// -*- C++ -*-
#ifndef __RACE_DETECT_UPDATE__
#define __RACE_DETECT_UPDATE__

#include "csan.h"
#include "shadow_mem.h"

// Check races on memory represented by this mem list with this read access.
// Once done checking, update the mem list with this new read access.
__attribute__((always_inline))
void check_races_and_update_with_read(const csi_id_t acc_id, MAType_t type,
                                      uintptr_t addr,
                                      size_t mem_size, bool on_stack,
                                      FrameData_t *f,
                                      Shadow_Memory &shadow_memory);

// Check races on memory represented by this mem list with this write access.
// Also, update the writers list.  Very similar to
// check_races_and_update_with_read function above.
__attribute__((always_inline))
void check_races_and_update_with_write(const csi_id_t acc_id, MAType_t type,
                                       uintptr_t addr,
                                       size_t mem_size, bool on_stack,
                                       FrameData_t *f,
                                       Shadow_Memory &shadow_memory);

// Check races on memory represented by this mem list with this mem access.
// Once done checking, update the mem list with the new mem access.
//
// is_read: whether this access is a read or not
// in_user_context: whether this access is made by user strand or runtime
//                  strand (e.g., update / reduce)
// inst_addr: the instruction that performs the read
// addr: the actual memory location accessed
// mem_size: the size of this memory access
// on_stack: whether this access is accessing a memory allocated on stack
// curr_sbag: the SBag of the current function context
// curr_top_pbag: the top-most PBag of the current function context
// curr_view_id: the view id of the currently executing strand, which is the
//               same as the view_id stored in the curr_top_pbag, but since we
//               create PBag lazily, the curr_top_pbag may be NULL.
template<bool is_read>
__attribute__((always_inline)) void
check_races_and_update(const csi_id_t acc_id, MAType_t type,
                       uintptr_t addr, size_t mem_size, bool on_stack,
                       FrameData_t *f, Shadow_Memory &shadow_memory);

#endif // __RACE_DETECT_UPDATE__
