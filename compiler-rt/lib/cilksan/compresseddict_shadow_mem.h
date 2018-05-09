// -*- C++ -*-
#ifndef __COMPRESSEDDICT_SHADOW_MEM__
#define __COMPRESSEDDICT_SHADOW_MEM__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>

#include <execinfo.h>
#include <inttypes.h>

#include "cilksan_internal.h"
#include "debug_util.h"
#include "disjointset.h"
#include "frame_data.h"
#include "mem_access.h"
#include "shadow_mem.h"
#include "spbag.h"
#include "stack.h"
// #include "run_dictionary.h"
// #include "lz_dictionary.h"
#include "static_dictionary.h"

class CompressedDictShadowMem : public ShadowMemoryType {

private:
  // address, function
  typedef std::map<uint64_t, value_type00>::iterator ShadowMemIter_t;

  Dictionary *my_read_dict;
  Dictionary *my_write_dict;
  Dictionary *my_alloc_dict;

  value_type00 *find(bool is_read, uintptr_t addr);
  value_type00 *find_group(bool is_read, uintptr_t addr, size_t max_size,
                           size_t &num_elems);
  value_type00 *find_exact_group(bool is_read, uintptr_t addr,
                                 size_t max_size, size_t &num_elems);
  // int element_count;
  void update(bool with_read, const csi_id_t acc_id,
              uintptr_t addr, size_t mem_size, bool on_stack, FrameData_t *f,
              const call_stack_t &call_stack);
  void check_race(bool prev_read, bool is_read, const csi_id_t acc_id,
                  uintptr_t addr, size_t mem_size,
                  bool on_stack, FrameData_t *f,
                  const call_stack_t &call_stack);

public:
  CompressedDictShadowMem();
  ~CompressedDictShadowMem() {destruct();}
  void insert_access(bool is_read, const csi_id_t acc_id,
                     uintptr_t addr, size_t mem_size, FrameData_t *f,
                     const call_stack_t &call_stack);
  void insert_access_into_group(bool is_read, const csi_id_t acc_id,
                                uintptr_t addr,
                                size_t mem_size, FrameData_t *f,
                                const call_stack_t &call_stack,
                                value_type00 *dst);
  bool does_access_exists(bool is_read, uintptr_t addr, size_t mem_size);
  void clear(size_t start, size_t size);
  void record_alloc(size_t start, size_t size,
                    FrameData_t *f,
                    const call_stack_t &call_stack,
                    csi_id_t alloca_id);
  void check_race_with_prev_read(const csi_id_t acc_id,
                                 uintptr_t addr, size_t mem_size, bool on_stack,
                                 FrameData_t *f,
                                 const call_stack_t &call_stack);
  void check_race_with_prev_write(bool is_read, const csi_id_t acc_id,
                                  uintptr_t addr,
                                  size_t mem_size, bool on_stack,
                                  FrameData_t *f,
                                  const call_stack_t &call_stack);
  void update_with_write(const csi_id_t acc_id,
                         uintptr_t addr, size_t mem_size, bool on_stack,
                         FrameData_t *f, const call_stack_t &call_stack);
  void update_with_read(const csi_id_t acc_id,
                        uintptr_t addr, size_t mem_size, bool on_stack,
                        FrameData_t *f, const call_stack_t &call_stack);
  void check_and_update_write(const csi_id_t acc_id,
                              uintptr_t addr, size_t mem_size, bool on_stack,
                              FrameData_t *f, const call_stack_t &call_stack);
  void destruct();

};

#endif // __COMPRESSEDDICT_SHADOW_MEM__
