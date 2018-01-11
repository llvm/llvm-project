// -*- C++ -*-
#ifndef __RACE_DETECT_UPDATE__
#define __RACE_DETECT_UPDATE__

#include "csan.h"
#include "shadow_mem.h"

// Check races on memory represented by this mem list with this read access.
// Once done checking, update the mem list with this new read access.
void check_races_and_update_with_read(const csi_id_t acc_id,
                                      uintptr_t addr,
                                      size_t mem_size, bool on_stack,
                                      FrameData_t *f,
                                      const call_stack_t &call_stack,
                                      Shadow_Memory &shadow_memory) {
  shadow_memory.update_with_read(acc_id, addr, mem_size, on_stack,
                                 f, call_stack);
  shadow_memory.check_race_with_prev_write(true, acc_id, addr,
                                           mem_size, on_stack, f, call_stack);
}

// Check races on memory represented by this mem list with this write access.
// Also, update the writers list.  Very similar to
// check_races_and_update_with_read function above.
void check_races_and_update_with_write(const csi_id_t acc_id,
                                       uintptr_t addr,
                                       size_t mem_size, bool on_stack,
                                       FrameData_t *f,
                                       const call_stack_t &call_stack,
                                       Shadow_Memory& shadow_memory) {
  // shadow_memory.check_race_with_prev_write(false, acc_id, inst_addr, addr,
  //                                          mem_size, on_stack, f,
  //                                          call_stack);
  // shadow_memory.update_with_write(acc_id, inst_addr, addr, mem_size, on_stack,
  //                                 f, call_stack);
  shadow_memory.check_and_update_write(acc_id, addr, mem_size,
                                       on_stack, f, call_stack);
  shadow_memory.check_race_with_prev_read(acc_id, addr, mem_size,
                                          on_stack, f, call_stack);
}

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
inline void
check_races_and_update(bool is_read, const csi_id_t acc_id,
                       uintptr_t addr, size_t mem_size, bool on_stack,
                       FrameData_t *f, const call_stack_t &call_stack,
                       Shadow_Memory& shadow_memory) {
  if (is_read)
    check_races_and_update_with_read(acc_id, addr, mem_size,
                                     on_stack, f, call_stack, shadow_memory);
  else
    check_races_and_update_with_write(acc_id, addr, mem_size,
                                      on_stack, f, call_stack, shadow_memory);
}

#endif // __RACE_DETECT_UPDATE__
