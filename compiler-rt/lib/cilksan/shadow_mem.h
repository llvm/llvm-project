// -*- C++ -*-
#ifndef __SHADOW_MEM__
#define __SHADOW_MEM__

#include <iostream>

#include "csan.h"
#include "debug_util.h"
#include "frame_data.h"
#include "race_info.h"

class ShadowMemoryType {
public:
  virtual void insert_access(bool is_read, const csi_id_t acc_id,
                             uintptr_t addr,
                             size_t mem_size, FrameData_t *f,
                             const call_stack_t &call_stack) = 0;

  virtual bool does_access_exists(bool is_read, uintptr_t addr,
                                  size_t mem_size) = 0;

  virtual void clear(size_t start, size_t size) = 0;

  virtual void record_alloc(size_t start, size_t size, FrameData_t *f,
                            const call_stack_t &call_stack, csi_id_t alloca_id) = 0;

  virtual void clear_alloc(size_t start, size_t size) = 0;

  virtual void check_race_with_prev_read(const csi_id_t acc_id,
                                         uintptr_t addr,
                                         size_t mem_size, bool on_stack,
                                         FrameData_t *f,
                                         const call_stack_t &call_stack) = 0;

  virtual void check_race_with_prev_write(bool is_read, const csi_id_t acc_id,
                                          uintptr_t addr,
                                          size_t mem_size, bool on_stack,
                                          FrameData_t *f,
                                          const call_stack_t &call_stack) = 0;

  virtual void update_with_write(const csi_id_t acc_id,
                                 uintptr_t addr, size_t mem_size,
                                 bool on_stack, FrameData_t *f,
                                 const call_stack_t &call_stack) = 0;

  virtual void update_with_read(const csi_id_t acc_id,
                                uintptr_t addr, size_t mem_size,
                                bool on_stack, FrameData_t *f,
                                const call_stack_t &call_stack) = 0;

  virtual void check_and_update_write(const csi_id_t acc_id,
                                      uintptr_t addr, size_t mem_size,
                                      bool on_stack, FrameData_t *f,
                                      const call_stack_t &call_stack) = 0;

  virtual void destruct() = 0;

  virtual ~ShadowMemoryType() = 0;
};

inline ShadowMemoryType::~ShadowMemoryType() {}

//  to be performance-engineered later
class Shadow_Memory {
  short type; // hash
  ShadowMemoryType* shadow_mem;

public:
  void test(){}

  void init();

  // Inserts access, and replaces any that are already in the shadow memory.
  void insert_access(bool is_read, const csi_id_t acc_id,
                     uintptr_t addr, size_t mem_size, FrameData_t *f,
                     const call_stack_t &call_stack) {
    shadow_mem->insert_access(is_read, acc_id, addr, mem_size, f, call_stack);
  }

  // Returns true if ANY bytes between addr and addr+mem_size are in the shadow
  // memory.
  bool does_access_exists (bool is_read, uintptr_t addr, size_t mem_size){
    return shadow_mem->does_access_exists(is_read, addr, mem_size);
  }

  void clear(size_t start, size_t size) {
    shadow_mem->clear(start, size);
  }

  void record_alloc(size_t start, size_t size, FrameData_t *f,
                    const call_stack_t &call_stack, csi_id_t alloca_id) {
    shadow_mem->record_alloc(start, size, f, call_stack, alloca_id);
  }

  void clear_alloc(size_t start, size_t size) {
    shadow_mem->clear_alloc(start, size);
  }

  void check_race_with_prev_read(const csi_id_t acc_id,
                                 uintptr_t addr, size_t mem_size, bool on_stack,
                                 FrameData_t *f,
                                 const call_stack_t &call_stack) {
    shadow_mem->check_race_with_prev_read(acc_id, addr, mem_size,
                                          on_stack, f, call_stack);
  }

  void check_race_with_prev_write(bool is_read, const csi_id_t acc_id,
                                  uintptr_t addr,
                                  size_t mem_size, bool on_stack,
                                  FrameData_t *f,
                                  const call_stack_t &call_stack) {
    shadow_mem->check_race_with_prev_write(is_read, acc_id, addr,
                                           mem_size, on_stack, f,
                                           call_stack);
  }

  void update_with_write(const csi_id_t acc_id,
                         uintptr_t addr, size_t mem_size, bool on_stack,
                         FrameData_t *f, const call_stack_t &call_stack) {
    shadow_mem->update_with_write(acc_id, addr, mem_size, on_stack,
                                  f, call_stack);
  }

  void update_with_read(const csi_id_t acc_id,
                        uintptr_t addr, size_t mem_size, bool on_stack,
                        FrameData_t *f, const call_stack_t &call_stack) {
    shadow_mem->update_with_read(acc_id, addr, mem_size, on_stack,
                                 f, call_stack);
  }

  void check_and_update_write(const csi_id_t acc_id,
                              uintptr_t addr, size_t mem_size, bool on_stack,
                              FrameData_t *f, const call_stack_t &call_stack) {
    shadow_mem->check_and_update_write(acc_id, addr, mem_size, on_stack,
                                       f, call_stack);
  }

  void destruct() {
    if (shadow_mem) {
      delete shadow_mem;
      shadow_mem = nullptr;
    }
  }

  ~Shadow_Memory() {destruct();}
};

#endif // __SHADOW_MEM__
