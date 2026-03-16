/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef DEVICE_AMD_HSA_H
#define DEVICE_AMD_HSA_H

#include <stdint.h>

#define DEVICE_COMPILER
#define LITTLEENDIAN_CPU
#include "hsa.h"
#include "amd_hsa_common.h"
#include "amd_hsa_elf.h"
#include "amd_hsa_kernel_code.h"
#include "amd_hsa_queue.h"
#include "amd_hsa_signal.h"
#undef DEVICE_COMPILER

#endif // DEVICE_AMD_HSA_H
