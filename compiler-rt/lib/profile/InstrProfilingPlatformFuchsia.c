/*===- InstrProfilingPlatformFuchsia.c - Profile data Fuchsia platform ----===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/
/*
 * This file implements the profiling runtime for Fuchsia and defines the
 * shared profile runtime interface. Each module (executable or DSO) statically
 * links in the whole profile runtime to satisfy the calls from its
 * instrumented code. Several modules in the same program might be separately
 * compiled and even use different versions of the instrumentation ABI and data
 * format. All they share in common is the VMO and the offset, which live in
 * exported globals so that exactly one definition will be shared across all
 * modules. Each module has its own independent runtime that registers its own
 * atexit hook to append its own data into the shared VMO which is published
 * via the data sink hook provided by Fuchsia's dynamic linker.
 */

#if defined(__Fuchsia__)

#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>

#include <zircon/process.h>
#include <zircon/sanitizer.h>
#include <zircon/status.h>
#include <zircon/syscalls.h>

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingUtil.h"

COMPILER_RT_VISIBILITY unsigned lprofRuntimeCounterRelocation(void) {
  return 1;
}

/* VMO that contains the profile data for this module. */
static zx_handle_t __llvm_profile_vmo;
/* Current offset within the VMO where data should be written next. */
static uint64_t __llvm_profile_offset;

static const char ProfileSinkName[] = "llvm-profile";

static inline void lprofWrite(const char *fmt, ...) {
  char s[256];

  va_list ap;
  va_start(ap, fmt);
  int ret = vsnprintf(s, sizeof(s), fmt, ap);
  va_end(ap);

  __sanitizer_log_write(s, ret + 1);
}

static uint32_t lprofVMOWriter(ProfDataWriter *This, ProfDataIOVec *IOVecs,
                               uint32_t NumIOVecs) {
  /* Compute the total length of data to be written. */
  size_t Length = 0;
  for (uint32_t I = 0; I < NumIOVecs; I++)
    Length += IOVecs[I].ElmSize * IOVecs[I].NumElm;

  /* Resize the VMO to ensure there's sufficient space for the data. */
  zx_status_t Status =
      _zx_vmo_set_size(__llvm_profile_vmo, __llvm_profile_offset + Length);
  if (Status != ZX_OK)
    return -1;

  /* Copy the data into VMO. */
  for (uint32_t I = 0; I < NumIOVecs; I++) {
    size_t Length = IOVecs[I].ElmSize * IOVecs[I].NumElm;
    if (IOVecs[I].Data) {
      Status = _zx_vmo_write(__llvm_profile_vmo, IOVecs[I].Data,
                             __llvm_profile_offset, Length);
      if (Status != ZX_OK)
        return -1;
    } else if (IOVecs[I].UseZeroPadding) {
      /* Resizing the VMO should zero fill. */
    }
    __llvm_profile_offset += Length;
  }

  /* Record the profile size as a property of the VMO. */
  _zx_object_set_property(__llvm_profile_vmo, ZX_PROP_VMO_CONTENT_SIZE,
                          &__llvm_profile_offset,
                          sizeof(__llvm_profile_offset));

  return 0;
}

static void initVMOWriter(ProfDataWriter *This) {
  This->Write = lprofVMOWriter;
  This->WriterCtx = NULL;
}

/* This method is invoked by the runtime initialization hook
 * InstrProfilingRuntime.o if it is linked in. */
COMPILER_RT_VISIBILITY
void __llvm_profile_initialize(void) {
  /* This symbol is defined as weak and initialized to -1 by the runtimer, but
   * compiler will generate a strong definition initialized to 0 when runtime
   * counter relocation is used. */
  if (__llvm_profile_counter_bias == -1) {
    lprofWrite("LLVM Profile: counter relocation at runtime is required\n");
    return;
  }

  /* Don't create VMO if it has been alread created. */
  if (__llvm_profile_vmo != ZX_HANDLE_INVALID) {
    lprofWrite("LLVM Profile: VMO has already been created\n");
    return;
  }

  const __llvm_profile_data *DataBegin = __llvm_profile_begin_data();
  const __llvm_profile_data *DataEnd = __llvm_profile_end_data();
  const uint64_t DataSize = __llvm_profile_get_data_size(DataBegin, DataEnd);
  const uint64_t CountersOffset = sizeof(__llvm_profile_header) +
    (DataSize * sizeof(__llvm_profile_data));

  zx_status_t Status;

  /* Create VMO to hold the profile data. */
  Status = _zx_vmo_create(0, ZX_VMO_RESIZABLE, &__llvm_profile_vmo);
  if (Status != ZX_OK) {
    lprofWrite("LLVM Profile: cannot create VMO: %s\n",
               _zx_status_get_string(Status));
    return;
  }

  /* Give the VMO a name that includes the module signature. */
  char VmoName[ZX_MAX_NAME_LEN];
  snprintf(VmoName, sizeof(VmoName), "%" PRIu64 ".profraw",
           lprofGetLoadModuleSignature());
  _zx_object_set_property(__llvm_profile_vmo, ZX_PROP_NAME, VmoName,
                          strlen(VmoName));

  /* Duplicate the handle since __sanitizer_publish_data consumes it. */
  zx_handle_t Handle;
  Status =
      _zx_handle_duplicate(__llvm_profile_vmo, ZX_RIGHT_SAME_RIGHTS, &Handle);
  if (Status != ZX_OK) {
    lprofWrite("LLVM Profile: cannot duplicate VMO handle: %s\n",
               _zx_status_get_string(Status));
    _zx_handle_close(__llvm_profile_vmo);
    __llvm_profile_vmo = ZX_HANDLE_INVALID;
    return;
  }

  /* Publish the VMO which contains profile data to the system. */
  __sanitizer_publish_data(ProfileSinkName, Handle);

  /* Use the dumpfile symbolizer markup element to write the name of VMO. */
  lprofWrite("LLVM Profile: {{{dumpfile:%s:%s}}}\n", ProfileSinkName, VmoName);

  /* Check if there is llvm/runtime version mismatch. */
  if (GET_VERSION(__llvm_profile_get_version()) != INSTR_PROF_RAW_VERSION) {
    lprofWrite("LLVM Profile: runtime and instrumentation version mismatch: "
               "expected %d, but got %d\n",
               INSTR_PROF_RAW_VERSION,
               (int)GET_VERSION(__llvm_profile_get_version()));
    return;
  }

  /* Write the profile data into the mapped region. */
  ProfDataWriter VMOWriter;
  initVMOWriter(&VMOWriter);
  if (lprofWriteData(&VMOWriter, 0, 0) != 0) {
    lprofWrite("LLVM Profile: failed to write data\n");
    return;
  }

  uint64_t Len = 0;
  Status = _zx_vmo_get_size(__llvm_profile_vmo, &Len);
  if (Status != ZX_OK) {
    lprofWrite("LLVM Profile: failed to get the VMO size: %s\n",
               _zx_status_get_string(Status));
    return;
  }

  uintptr_t Mapping;
  Status =
      _zx_vmar_map(_zx_vmar_root_self(), ZX_VM_PERM_READ | ZX_VM_PERM_WRITE,
                   0, __llvm_profile_vmo, 0, Len, &Mapping);
  if (Status != ZX_OK) {
    lprofWrite("LLVM Profile: failed to map the VMO: %s\n",
               _zx_status_get_string(Status));
    return;
  }

  /* Update the profile fields based on the current mapping. */
  __llvm_profile_counter_bias = (intptr_t)Mapping -
      (uintptr_t)__llvm_profile_begin_counters() + CountersOffset;
}

#endif
