/* -*- Mode: C++ -*- */

#ifndef _MEM_ACCESS_H
#define _MEM_ACCESS_H

#include <iostream>
#include <inttypes.h>

#include "cilksan_internal.h"
#include "debug_util.h"
#include "disjointset.h"
#include "spbag.h"

// macro for address manipulation for shadow mem
// used in all shadow memory implementations
#define ADDR_TO_KEY(addr) ((uint64_t) ((uint64_t)addr >> 3))

#define MAX_GRAIN_SIZE 64
// #define MAX_GRAIN_SIZE 8
// a mask that keeps all the bits set except for the least significant bits
// that represent the max grain size
#define MAX_GRAIN_MASK (~(uintptr_t)(MAX_GRAIN_SIZE-1))

#define MALLOC_ALIGN_SIZE 1
#define MALLOC_ALIGN_MASK (~(size_t)(MALLOC_ALIGN_SIZE-1))
#define ALIGN_FOR_MALLOC(size) \
  ((size_t) ((size + (MALLOC_ALIGN_SIZE-1)) & MALLOC_ALIGN_MASK))

// If the value is already divisible by MAX_GRAIN_SIZE, return the value;
// otherwise return the previous / next value divisible by MAX_GRAIN_SIZE.
#define ALIGN_BY_PREV_MAX_GRAIN_SIZE(addr) ((uintptr_t) (addr & MAX_GRAIN_MASK))
#define ALIGN_BY_NEXT_MAX_GRAIN_SIZE(addr) \
  ((uintptr_t) ((addr+(MAX_GRAIN_SIZE-1)) & MAX_GRAIN_MASK))

enum GrainType_t { UNINIT = -1, ONE = 0, TWO = 1, FOUR = 2, EIGHT = 3 };
static const int gtype_to_mem_size[4] = { 1, 2, 4, 8 };
#define MAX_GTYPE EIGHT // the max value that the enum GrainType_t can take

// check if addr is aligned with the granularity represented by gtype
#define IS_ALIGNED_WITH_GTYPE(addr, gtype) \
  ((addr & ((uint64_t)gtype_to_mem_size[gtype]-1)) == 0)

// Instantiation declarations for functions and static members declared in
// cilksan.cpp.
template<>
void (*DisjointSet_t<SPBagInterface *>::dtor_callback)(DisjointSet_t *);
template<>
DisjointSet_t<SPBagInterface *> *DisjointSet_t<SPBagInterface *>::free_list;

// Struct to hold a pair of disjoint sets corresponding to the last reader and writer
typedef struct MemAccess_t {

  // the containing function of this access
  DisjointSet_t<SPBagInterface *> *func;
  uint64_t rip; // the instruction address of this access
  int16_t ref_count; // number of pointers aliasing to this object
  // ref_count == 0 if only a single unique pointer to this object exists

  MemAccess_t(DisjointSet_t<SPBagInterface *> *_func, uint64_t _rip)
    : func(_func), rip(_rip), ref_count(0)
  {
    func->inc_ref_count();
  }

  ~MemAccess_t() {
    func->dec_ref_count();
  }

  // NOTE: curr_pbag may be NULL because we create it lazily.
  inline bool races_with(uintptr_t addr, bool on_stack,
                         DisjointSet_t<SPBagInterface *> *curr_pbag) {
    bool has_race = false;
    // cilksan_assert(curr_vid != UNINIT_VIEW_ID);

    SPBagInterface *lca = func->get_set_node();
    // SPBagInterface *cur_node = func->get_node();
    // we are done if LCA is an S-node.
    if (lca->is_PBag()) {
      // if memory is allocated on stack, the accesses race with each other
      // only if the mem location is allocated in shared ancestor's stack
      // if stack_check = false, there is no race.
      // if stack_check = true, it's a race only if all other conditions apply.
      //bool stack_check = (!on_stack || addr >= cur_node->get_rsp());
      bool stack_check = (!on_stack || addr >= lca->get_rsp());
      has_race = stack_check;
      // if (cnt == USER) {
      //   has_race = stack_check;
      // } else {
      //   cilksan_assert(lca->get_view_id() != UNINIT_VIEW_ID);
      //   if(cnt == REDUCE) {
      //     // use the top_pbag's view id as the view id of the REDUCE strand
      //     curr_vid = curr_top_pbag->get_set_node()->get_view_id();
      //   }
      //   has_race = (lca->get_view_id() != curr_vid) && stack_check;
      // }
    }
    return has_race;
  }

  inline int16_t inc_ref_count() { ref_count++; return ref_count; }
  inline int16_t dec_ref_count() { ref_count--; return ref_count; }

  // for debugging use
  inline friend
  std::ostream& operator<<(std::ostream & ostr, MemAccess_t *acc) {
    ostr << "function: " << acc->func->get_node()->get_func_id();
    ostr << ", rip " << std::hex << "0x" << acc->rip;
    return ostr;
  }

} MemAccess_t;


class MemAccessList_t {

private:
  // the smallest addr of memory locations that this MemAccessList represents

  static inline enum GrainType_t mem_size_to_gtype(size_t size) {
    switch(size) {
      case 8:
        return EIGHT;
      case 4:
        return FOUR;
      case 2:
        return TWO;
      default: // everything else gets byte-granularity
        return ONE;
    }
  }

  // get the start and end indices and gtype to use for accesing the readers /
  // writers lists; the gtype is the largest granularity that this memory access
  // is aligned with
  static enum GrainType_t
  get_mem_index(uintptr_t addr, size_t size, int& start, int& end);

  // helper function: break one of the MemAcess list into a smaller granularity;
  // called by break_readers/writers_into_smaller_gtype.
  //
  // for_read: if true, break the readers list, otherwise break the writers
  // new_gtype: the desired granularity
  void break_list_into_smaller_gtype(bool for_read, enum GrainType_t new_gtype);

  inline void break_readers_into_smaller_gtype(enum GrainType_t new_gtype) {
    break_list_into_smaller_gtype(true, new_gtype);
  }

  inline void break_writers_into_smaller_gtype(enum GrainType_t new_gtype) {
    break_list_into_smaller_gtype(false, new_gtype);
  }

  // Check races on memory represented by this mem list with this read access
  // Once done checking, update the mem list with this new read access
  void check_races_and_update_with_read(uintptr_t inst_addr, uintptr_t addr,
					size_t mem_size, bool on_stack,
					DisjointSet_t<SPBagInterface *> *curr_sbag,
					DisjointSet_t<SPBagInterface *> *curr_pbag);

  // Check races on memory represented by this mem list with this write access
  // Also, update the writers list.  Very similar to
  // check_races_and_update_with_read function above.
  void check_races_and_update_with_write(uintptr_t inst_addr, uintptr_t addr,
                              size_t mem_size, bool on_stack,
                              DisjointSet_t<SPBagInterface *> *curr_sbag,
                              DisjointSet_t<SPBagInterface *> *curr_pbag);

public:

  static inline int get_prev_aligned_index(int index, enum GrainType_t gtype) {

    if( IS_ALIGNED_WITH_GTYPE(index, gtype) ) {
      return index;
    }
    return ((index >> gtype) << gtype);
  } //sk made public

  uint64_t start_addr;
  enum GrainType_t reader_gtype;
  MemAccess_t *readers[MAX_GRAIN_SIZE];
  enum GrainType_t writer_gtype;
  MemAccess_t *writers[MAX_GRAIN_SIZE]; //sk made public

  // Constructor.
  //
  // addr: the memory address of the access
  // is_read: whether the initializing memory access is a read
  // acc: the memory access that causes this MemAccessList_t to be created
  // mem_size: the size of the access
  MemAccessList_t(uintptr_t addr, bool is_read,
                  MemAccess_t *acc, size_t mem_size);

  ~MemAccessList_t();

  // Check races on memory represented by this mem list with this mem access
  // Once done checking, update the mem list with the new mem access
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
  check_races_and_update(bool is_read, uint64_t inst_addr, uint64_t addr,
                         size_t mem_size, bool on_stack,
                         DisjointSet_t<SPBagInterface *> *curr_sbag,
                         DisjointSet_t<SPBagInterface *> *curr_pbag) {
    if (is_read)
      check_races_and_update_with_read(inst_addr, addr, mem_size, on_stack,
                                       curr_sbag, curr_pbag);
    else
      check_races_and_update_with_write(inst_addr, addr, mem_size, on_stack,
                                        curr_sbag, curr_pbag);
  }

}; // end of class MemAccessList_t def

#endif // _MEM_ACCESS_H
