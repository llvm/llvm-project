/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/
#include "atmi_runtime.h"
#include "internal.h"
#include "rt.h"
#include <hsa.h>
#include <hsa_ext_amd.h>
#include <memory>

/*
 * Initialize/Finalize
 */
atmi_status_t atmi_init() { return core::Runtime::Initialize(); }

atmi_status_t atmi_finalize() { return core::Runtime::Finalize(); }

/*
 * Machine Info
 */
atmi_machine_t *atmi_machine_get_info() {
  return core::Runtime::GetMachineInfo();
}

/*
 * Modules
 */
atmi_status_t atmi_module_register_from_memory_to_place(
    void *module_bytes, size_t module_size, atmi_place_t place,
    atmi_status_t (*on_deserialized_data)(void *data, size_t size,
                                          void *cb_state),
    void *cb_state) {
  return core::Runtime::getInstance().RegisterModuleFromMemory(
      module_bytes, module_size, place, on_deserialized_data, cb_state);
}

/*
 * Data
 */

static hsa_status_t invoke_hsa_copy(hsa_signal_t sig, void *dest,
                                    const void *src, size_t size,
                                    hsa_agent_t agent) {
  const hsa_signal_value_t init = 1;
  const hsa_signal_value_t success = 0;
  hsa_signal_store_screlease(sig, init);

  hsa_status_t err =
      hsa_amd_memory_async_copy(dest, agent, src, agent, size, 0, NULL, sig);
  if (err != HSA_STATUS_SUCCESS) {
    return err;
  }

  // async_copy reports success by decrementing and failure by setting to < 0
  hsa_signal_value_t got = init;
  while (got == init) {
    got = hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_NE, init,
                                    UINT64_MAX, ATMI_WAIT_STATE);
  }

  if (got != success) {
    return HSA_STATUS_ERROR;
  }

  return err;
}

struct atmiFreePtrDeletor {
  void operator()(void *p) {
    atmi_free(p); // ignore failure to free
  }
};

atmi_status_t atmi_memcpy_h2d(hsa_signal_t signal, void *deviceDest,
                              const void *hostSrc, size_t size,
                              hsa_agent_t agent) {
  hsa_status_t rc = hsa_memory_copy(deviceDest, hostSrc, size);

  // hsa_memory_copy sometimes fails in situations where
  // allocate + copy succeeds. Looks like it might be related to
  // locking part of a read only segment. Fall back for now.
  if (rc == HSA_STATUS_SUCCESS) {
    return ATMI_STATUS_SUCCESS;
  }

  void *tempHostPtr;
  atmi_mem_place_t CPU = ATMI_MEM_PLACE_CPU_MEM(0, 0, 0);
  atmi_status_t ret = atmi_malloc(&tempHostPtr, size, CPU);
  if (ret != ATMI_STATUS_SUCCESS) {
    DEBUG_PRINT("atmi_malloc: Unable to alloc %d bytes for temp scratch\n",
                size);
    return ret;
  }
  std::unique_ptr<void, atmiFreePtrDeletor> del(tempHostPtr);
  memcpy(tempHostPtr, hostSrc, size);

  if (invoke_hsa_copy(signal, deviceDest, tempHostPtr, size, agent) !=
      HSA_STATUS_SUCCESS) {
    return ATMI_STATUS_ERROR;
  }
  return ATMI_STATUS_SUCCESS;
}

atmi_status_t atmi_memcpy_d2h(hsa_signal_t signal, void *dest,
                              const void *deviceSrc, size_t size,
                              hsa_agent_t agent) {
  hsa_status_t rc = hsa_memory_copy(dest, deviceSrc, size);

  // hsa_memory_copy sometimes fails in situations where
  // allocate + copy succeeds. Looks like it might be related to
  // locking part of a read only segment. Fall back for now.
  if (rc == HSA_STATUS_SUCCESS) {
    return ATMI_STATUS_SUCCESS;
  }

  void *tempHostPtr;
  atmi_mem_place_t CPU = ATMI_MEM_PLACE_CPU_MEM(0, 0, 0);
  atmi_status_t ret = atmi_malloc(&tempHostPtr, size, CPU);
  if (ret != ATMI_STATUS_SUCCESS) {
    DEBUG_PRINT("atmi_malloc: Unable to alloc %d bytes for temp scratch\n",
                size);
    return ret;
  }
  std::unique_ptr<void, atmiFreePtrDeletor> del(tempHostPtr);

  if (invoke_hsa_copy(signal, tempHostPtr, deviceSrc, size, agent) !=
      HSA_STATUS_SUCCESS) {
    return ATMI_STATUS_ERROR;
  }

  memcpy(dest, tempHostPtr, size);
  return ATMI_STATUS_SUCCESS;
}

atmi_status_t atmi_free(void *ptr) { return core::Runtime::Memfree(ptr); }

atmi_status_t atmi_malloc(void **ptr, size_t size, atmi_mem_place_t place) {
  return core::Runtime::Malloc(ptr, size, place);
}
