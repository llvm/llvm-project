#include <memory>

#include "compresseddict_shadow_mem.h"

size_t MemoryAccess_t::FreeNode_t::FreeNode_ObjSize = 0;
MemoryAccess_t::FreeNode_t *MemoryAccess_t::free_list = nullptr;

extern CilkSanImpl_t CilkSanImpl;

CompressedDictShadowMem::CompressedDictShadowMem() {
    int dict_type = 2;
    // element_count = 0;
    switch (dict_type) {
        // case 0:
        //     my_read_dict = new Run_Dictionary();
        //     my_write_dict = new Run_Dictionary();
        //     break;
        // case 1:
        //     my_read_dict = new LZ_Dictionary();
        //     my_write_dict = new LZ_Dictionary();
        //     break;
        case 2:
            my_read_dict = new Static_Dictionary();
            my_write_dict = new Static_Dictionary();
            my_alloc_dict = new Static_Dictionary();
            break;
    }
}

void CompressedDictShadowMem::insert_access(bool is_read, const csi_id_t acc_id,
                                            uintptr_t addr,
                                            size_t mem_size, FrameData_t *f,
                                            const call_stack_t &call_stack) {
  // element_count += mem_size;
  uint64_t key = addr;
  // std::cerr << __PRETTY_FUNCTION__ << " key = 0x" << std::hex << key << "\n";
  if (is_read) {
    // my_read_dict->erase(key, mem_size);
    // my_read_dict->insert(key, mem_size,
    //                      MemoryAccess_t(f->Sbag, // inst_addr,
    //                                     acc_id, call_stack));
    my_read_dict->set(key, mem_size,
                      MemoryAccess_t(f->Sbag, acc_id, call_stack));
  } else {
    // my_write_dict->erase(key, mem_size);
    // my_write_dict->insert(key, mem_size,
    //                       MemoryAccess_t(f->Sbag, // inst_addr,
    //                                      acc_id, call_stack));
    my_write_dict->set(key, mem_size,
                       MemoryAccess_t(f->Sbag, acc_id, call_stack));
  }
}

void CompressedDictShadowMem::insert_access_into_group(bool is_read, const csi_id_t acc_id,
                                                       uintptr_t addr,
                                                       size_t mem_size, FrameData_t *f,
                                                       const call_stack_t &call_stack,
                                                       value_type00 *dst) {
  // element_count += mem_size;
  uint64_t key = addr;
  // std::cerr << __PRETTY_FUNCTION__ << " key = 0x" << std::hex << key << "\n";
  if (is_read) {
    // my_read_dict->erase(key, mem_size);
    my_read_dict->insert_into_found_group(key, mem_size, dst,
                                          MemoryAccess_t(f->Sbag, acc_id,
                                                         call_stack));
  } else {
    // my_write_dict->erase(key, mem_size);
    my_write_dict->insert_into_found_group(key, mem_size, dst,
                                           MemoryAccess_t(f->Sbag, acc_id,
                                                          call_stack));
  }
}

// given an address in the (hash) shadow memory, return the mem-access object
// for that address (or null if the address is not in the shadow memory)
value_type00 *CompressedDictShadowMem::find(bool is_read, uintptr_t addr) {
  if (is_read)
    return my_read_dict->find(addr);
  else
    return my_write_dict->find(addr);
}

value_type00 *CompressedDictShadowMem::find_group(bool is_read, uintptr_t addr,
                                                  size_t max_size,
                                                  size_t &num_elems) {
  if (is_read)
    return my_read_dict->find_group(addr, max_size, num_elems);
  else
    return my_write_dict->find_group(addr, max_size, num_elems);
}

value_type00 *CompressedDictShadowMem::find_exact_group(bool is_read,
                                                        uintptr_t addr,
                                                        size_t max_size,
                                                        size_t &num_elems) {
  if (is_read)
    return my_read_dict->find_exact_group(addr, max_size, num_elems);
  else
    return my_write_dict->find_exact_group(addr, max_size, num_elems);
}

bool CompressedDictShadowMem::does_access_exists(bool is_read, uintptr_t addr,
                                                 size_t mem_size) {
  if (is_read)
    return my_read_dict->includes(addr, mem_size);
  else
    return my_write_dict->includes(addr, mem_size);
}

void CompressedDictShadowMem::clear(size_t start, size_t size) {
  // my_read_dict->erase(start, end - start);
  // my_write_dict->erase(start, end - start);
  my_read_dict->erase(start, size);
  my_write_dict->erase(start, size);
  //while(start != end) {
  //    uint64_t key = start;
  //    my_read_dict->erase(key);
  //    my_write_dict->erase(key);
  //
  //    start++;
  //}
}

void CompressedDictShadowMem::record_alloc(size_t start, size_t size,
                                           FrameData_t *f,
                                           const call_stack_t &call_stack,
                                           csi_id_t alloca_id) {
  my_alloc_dict->set(start, size, MemoryAccess_t(f->Sbag, alloca_id, call_stack));
}

// prev_read: are we checking with previous reads or writes?
// is_read: is the current access read or write?
void CompressedDictShadowMem::check_race(bool prev_read, bool is_read,
                                         const csi_id_t acc_id,
                                         uintptr_t addr,
                                         size_t mem_size, bool on_stack,
                                         FrameData_t *f,
                                         const call_stack_t &call_stack) {
  // std::cerr << __PRETTY_FUNCTION__ << " prev_read " << prev_read << " is_read " << is_read << "\n";
  size_t index = 0;
  while (index < mem_size) {
    uintptr_t shifted_addr = addr + index;
    size_t current_size = 0;
    value_type00 *access = find_group(prev_read, shifted_addr, mem_size - index,
                                      current_size);
    assert(current_size > 0);
    assert(shifted_addr + current_size <= addr + mem_size);
    if (access && access->isValid()) {
      auto func = access->getFunc();
      // bool has_race = false;
      cilksan_assert(func);
      // cilksan_assert(curr_vid != UNINIT_VIEW_ID);

      SPBagInterface *lca = func->get_set_node();
      // SPBagInterface *cur_node = func->get_node();
      if (lca->is_PBag()) {
        // If memory is allocated on stack, the accesses race with each other
        // only if the mem location is allocated in shared ancestor's stack.  We
        // don't need to check for this because we clear shadow memory;
        // non-shared stack can't race because earlier one would've been cleared

        auto alloc_find = my_alloc_dict->find(shifted_addr);
        AccessLoc_t alloc_access =
          (alloc_find == nullptr) ? AccessLoc_t() : alloc_find->getLoc();
        if (prev_read) // checking the current access with previous reads
          CilkSanImpl.report_race(
              access->getLoc(), AccessLoc_t(acc_id, call_stack), alloc_access,
              shifted_addr, RW_RACE);
        else {  // check the current access with previous writes
          if (is_read) // the current access is a read
            CilkSanImpl.report_race(
                access->getLoc(), AccessLoc_t(acc_id, call_stack), alloc_access,
                shifted_addr, WR_RACE);
          else
            CilkSanImpl.report_race(
                access->getLoc(), AccessLoc_t(acc_id, call_stack), alloc_access,
                shifted_addr, WW_RACE);
        }
      }
    }
    index += current_size;
  }
}

void CompressedDictShadowMem::check_race_with_prev_read(const csi_id_t acc_id,
                                                        uintptr_t addr,
                                                        size_t mem_size,
                                                        bool on_stack,
                                                        FrameData_t *f,
                                                        const call_stack_t &call_stack) {
  // the second argument does not matter here
  check_race(true, false, acc_id, addr, mem_size, on_stack, f, call_stack);
}

void CompressedDictShadowMem::check_race_with_prev_write(bool is_read,
                                                         const csi_id_t acc_id,
                                                         uintptr_t addr,
                                                         size_t mem_size,
                                                         bool on_stack,
                                                         FrameData_t *f,
                                                         const call_stack_t &call_stack) {
  check_race(false, is_read, acc_id, addr, mem_size, on_stack, f, call_stack);
}

void CompressedDictShadowMem::update(bool with_read, const csi_id_t acc_id,
                                     uintptr_t addr,
                                     size_t mem_size, bool on_stack,
                                     FrameData_t *f,
                                     const call_stack_t &call_stack) {
  size_t index = 0;
  while (index < mem_size) {
    uintptr_t shifted_addr = addr + index;
    size_t current_size = 0;
    value_type00 *access = find_exact_group(with_read, shifted_addr,
                                            mem_size - index, current_size);
    assert(current_size > 0);
    assert(shifted_addr + current_size <= addr + mem_size);
    if (!access || !access->isValid())
      insert_access_into_group(with_read, acc_id, shifted_addr,
                               current_size, f, call_stack, access);
    else {
      auto func = access->getFunc();
      SPBagInterface *last_rset = func->get_set_node();
      // replace it only if it is in series with this access, i.e., if it's
      // one of the following:
      // a) in a SBag
      // b) in a PBag but should have been replaced because the access is
      // actually on the newly allocated stack frame (i.e., cactus stack abstraction)
      // // c) access is made by a REDUCE strand and previous access is in the
      // // top-most PBag.
      if (last_rset->is_SBag() ||
          (on_stack && last_rset->get_rsp() >= shifted_addr)) {
        // func->dec_ref_count(current_size);
        // access->dec_AccessLoc_ref_count(current_size);
        access->dec_ref_counts(current_size);
        // note that ref count is decremented regardless
        insert_access_into_group(with_read, acc_id, shifted_addr,
                                 current_size, f, call_stack, access);
      }
    }
    index += current_size;
  }
}

void CompressedDictShadowMem::update_with_write(const csi_id_t acc_id,
                                                uintptr_t addr, size_t mem_size,
                                                bool on_stack, FrameData_t *f,
                                                const call_stack_t &call_stack) {
  update(false, acc_id, addr, mem_size, on_stack, f, call_stack);
}

void CompressedDictShadowMem::update_with_read(const csi_id_t acc_id,
                                               uintptr_t addr, size_t mem_size,
                                               bool on_stack, FrameData_t *f,
                                               const call_stack_t &call_stack) {
  update(true, acc_id, addr, mem_size, on_stack, f, call_stack);
}

void CompressedDictShadowMem::check_and_update_write(
    const csi_id_t acc_id, uintptr_t addr, size_t mem_size,
    bool on_stack, FrameData_t *f, const call_stack_t &call_stack) {
  size_t index = 0;
  while (index < mem_size) {
    uintptr_t shifted_addr = addr + index;
    size_t current_size = 0;
    value_type00 *access = find_exact_group(false, shifted_addr,
                                            mem_size - index, current_size);
    assert(current_size > 0);
    assert(shifted_addr + current_size <= addr + mem_size);
    if (!access || !access->isValid())
      insert_access_into_group(false, acc_id, shifted_addr,
                               current_size, f, call_stack, access);
    else {
      auto func = access->getFunc();
      // bool has_race = false;
      cilksan_assert(func);
      // cilksan_assert(curr_vid != UNINIT_VIEW_ID);
      SPBagInterface *lca = func->get_set_node();

      // Check for races

      // SPBagInterface *cur_node = func->get_node();
      if (lca->is_PBag()) {
        auto alloc_find = my_alloc_dict->find(shifted_addr);
        AccessLoc_t alloc_access =
          (alloc_find == nullptr) ? AccessLoc_t() : alloc_find->getLoc();
        // check the current access with previous writes
        CilkSanImpl.report_race(
            access->getLoc(), AccessLoc_t(acc_id, call_stack), alloc_access,
            shifted_addr, WW_RACE);
      }

      // Update the table

      // replace it only if it is in series with this access, i.e., if it's
      // one of the following:
      // a) in a SBag
      // b) in a PBag but should have been replaced because the access is
      // actually on the newly allocated stack frame (i.e., cactus stack abstraction)
      // // c) access is made by a REDUCE strand and previous access is in the
      // // top-most PBag.
      if (lca->is_SBag() ||
          (on_stack && lca->get_rsp() >= shifted_addr)) {
        // func->dec_ref_count(current_size);
        // access->dec_AccessLoc_ref_count(current_size);
        access->dec_ref_counts(current_size);
        // note that ref count is decremented regardless
        insert_access_into_group(false, acc_id, shifted_addr,
                                 current_size, f, call_stack, access);
      }
    }
    index += current_size;
  }
}

void CompressedDictShadowMem::destruct() {
  //my_read_dict->destruct();
  //my_write_dict->destruct();
  delete my_read_dict;
  delete my_write_dict;
  delete my_alloc_dict;
  MemoryAccess_t::cleanup_freelist();
}
