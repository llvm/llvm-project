#include <inttypes.h>

#include "cilksan_internal.h"
#include "mem_access.h"

extern void report_race(uintptr_t first_inst, uintptr_t second_inst,
                        uintptr_t addr, enum RaceType_t race_type);

// get the start and end indices and gtype to use for accesing the readers /
// writers lists; the gtype is the largest granularity that this memory access
// is aligned with
enum GrainType_t
MemAccessList_t::get_mem_index(uintptr_t addr, size_t size,
                               int& start, int& end) {

  start = (int) (addr & (uintptr_t)(MAX_GRAIN_SIZE-1));
  end = (int) ((addr + size) & (uintptr_t)(MAX_GRAIN_SIZE-1));
  if (end == 0) end = MAX_GRAIN_SIZE;

  enum GrainType_t gtype = mem_size_to_gtype(size);
  if (IS_ALIGNED_WITH_GTYPE(addr, gtype) == false) { gtype = ONE; }

  return gtype;
}

void MemAccessList_t::break_list_into_smaller_gtype(bool for_read,
						    enum GrainType_t new_gtype) {


  MemAccess_t **l = writers;
  enum GrainType_t gtype = writer_gtype;
  if (for_read) {
    l = readers;
    gtype = reader_gtype;
  }
  const int stride = gtype_to_mem_size[new_gtype];
  MemAccess_t *acc = l[0];

  for (int i = stride; i < MAX_GRAIN_SIZE; i += stride) {
    if (IS_ALIGNED_WITH_GTYPE(i, gtype)) {
      acc = l[i];
    } else if(acc) {
      acc->inc_ref_count();
      l[i] = acc;
    }
  }

  if(for_read) {
    reader_gtype = new_gtype;
  } else {
    writer_gtype = new_gtype;
  }
}


MemAccessList_t::MemAccessList_t(uintptr_t addr, bool is_read,
                                 MemAccess_t *acc, size_t mem_size)
  : start_addr(ALIGN_BY_PREV_MAX_GRAIN_SIZE(addr)),
    reader_gtype(UNINIT), writer_gtype(UNINIT) {

  for (int i = 0; i < MAX_GRAIN_SIZE; i++) {
    readers[i] = writers[i] = NULL;
  }

  int start, end;
  const enum GrainType_t gtype = get_mem_index(addr, mem_size, start, end);

  MemAccess_t **l;
  if (is_read) {
    reader_gtype = gtype;
    l = readers;
  } else {
    writer_gtype = gtype;
    l = writers;
  }
  for (int i = start; i < end; i += gtype_to_mem_size[gtype]) {
    acc->inc_ref_count();
    l[i] = acc;
  }
}

MemAccessList_t::~MemAccessList_t() {
  MemAccess_t *acc;
  if (reader_gtype != UNINIT) {
    for (int i = 0; i < MAX_GRAIN_SIZE; i+=gtype_to_mem_size[reader_gtype]) {
      acc = readers[i];
      if(acc && acc->dec_ref_count() == 0) {
        delete acc;
        readers[i] = 0;
      }
    }
  }

  if(writer_gtype != UNINIT) {
    for(int i=0; i < MAX_GRAIN_SIZE; i+=gtype_to_mem_size[writer_gtype]) {
      acc = writers[i];
      if(acc && acc->dec_ref_count() == 0) {
        delete acc;
        writers[i] = 0;
      }
    }
  }
}

/*
#if CILKSAN_DEBUG
void MemAccessList_t::check_invariants(uint64_t current_func_id) {
  SPBagInterface *lca;
  for (int i = 0; i < MAX_GRAIN_SIZE; i++) {
    if (readers[i]) {
      lca = readers[i]->func->get_set_node();
      cilksan_assert(current_func_id >= lca->get_func_id());
      // if LCA is a P-node (Cilk function), its rsp must have been initialized
      cilksan_assert(lca->is_SBag() || lca->get_rsp() != UNINIT_STACK_PTR);
    }
    if (writers[i]) { // same checks for the writers
      lca = writers[i]->func->get_set_node();
      cilksan_assert(current_func_id >= lca->get_func_id());
      cilksan_assert(lca->is_SBag() || lca->get_rsp() != UNINIT_STACK_PTR);
    }
  }
}
#endif // CILKSAN_DEBUG
*/

