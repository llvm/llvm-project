// -*- C++ -*-
#ifndef __SHADOW_MEM__
#define __SHADOW_MEM__

#include "csan.h"
#include "debug_util.h"
#include "frame_data.h"

// class ShadowMemoryType {
// public:
//   virtual void insert_access(bool is_read, const csi_id_t acc_id,
//                              uintptr_t addr,
//                              size_t mem_size, FrameData_t *f,
//                              const call_stack_t &call_stack) = 0;

//   virtual bool does_access_exists(bool is_read, uintptr_t addr,
//                                   size_t mem_size) = 0;

//   virtual void clear(size_t start, size_t size) = 0;

//   virtual void record_alloc(size_t start, size_t size, FrameData_t *f,
//                             const call_stack_t &call_stack, csi_id_t alloca_id) = 0;

//   virtual void clear_alloc(size_t start, size_t size) = 0;

//   virtual void check_race_with_prev_read(const csi_id_t acc_id,
//                                          uintptr_t addr,
//                                          size_t mem_size, bool on_stack,
//                                          FrameData_t *f,
//                                          const call_stack_t &call_stack) = 0;

//   virtual void check_race_with_prev_write(const csi_id_t acc_id,
//                                           uintptr_t addr,
//                                           size_t mem_size, bool on_stack,
//                                           FrameData_t *f,
//                                   const call_stack_t &call_stack) = 0;

//   virtual void update_with_write(const csi_id_t acc_id,
//                                  uintptr_t addr, size_t mem_size,
//                                  bool on_stack, FrameData_t *f,
//                                  const call_stack_t &call_stack) = 0;

//   virtual void update_with_read(const csi_id_t acc_id,
//                                 uintptr_t addr, size_t mem_size,
//                                 bool on_stack, FrameData_t *f,
//                                 const call_stack_t &call_stack) = 0;

//   virtual void check_and_update_write(const csi_id_t acc_id,
//                                       uintptr_t addr, size_t mem_size,
//                                       bool on_stack, FrameData_t *f,
//                                       const call_stack_t &call_stack) = 0;

//   virtual void destruct() = 0;

//   virtual ~ShadowMemoryType() = 0;
// };

// inline ShadowMemoryType::~ShadowMemoryType() {}

// Forward declarations
class CilkSanImpl_t;
class SimpleShadowMem;

//  to be performance-engineered later
class Shadow_Memory {
  // short type; // hash
  // ShadowMemoryType* shadow_mem;
  SimpleShadowMem *shadow_mem;

public:
  ~Shadow_Memory() {destruct();}

  void init(CilkSanImpl_t &CilkSanImpl);

  // Inserts access, and replaces any that are already in the shadow memory.
  template<bool is_read>
  void insert_access(const csi_id_t acc_id, uintptr_t addr, size_t mem_size,
                     FrameData_t *f);

  // Returns true if ANY bytes between addr and addr+mem_size are in the shadow
  // memory.
  template<bool is_read>
  __attribute__((always_inline))
  bool does_access_exists(uintptr_t addr, size_t mem_size) const;

  __attribute__((always_inline))
  void clear(size_t start, size_t size);

  void record_alloc(size_t start, size_t size, FrameData_t *f,
                    csi_id_t alloca_id);

  void record_free(size_t start, size_t size, FrameData_t *f,
                   csi_id_t free_id, MAType_t type);

  void clear_alloc(size_t start, size_t size);

  __attribute__((always_inline))
  void check_race_with_prev_read(const csi_id_t acc_id, uintptr_t addr,
                                 size_t mem_size, bool on_stack,
                                 FrameData_t *f) const;

  template<bool is_read>
  __attribute__((always_inline))
  void check_race_with_prev_write(const csi_id_t acc_id, MAType_t type,
                                  uintptr_t addr, size_t mem_size,
                                  bool on_stack, FrameData_t *f) const;

  __attribute__((always_inline))
  void update_with_write(const csi_id_t acc_id, MAType_t type, uintptr_t addr,
                         size_t mem_size, bool on_stack, FrameData_t *f);

  __attribute__((always_inline))
  void update_with_read(const csi_id_t acc_id, uintptr_t addr, size_t mem_size,
                        bool on_stack, FrameData_t *f);

  __attribute__((always_inline))
  void check_and_update_write(const csi_id_t acc_id, MAType_t type,
                              uintptr_t addr, size_t mem_size, bool on_stack,
                              FrameData_t *f);

  void destruct();
};

#endif // __SHADOW_MEM__
