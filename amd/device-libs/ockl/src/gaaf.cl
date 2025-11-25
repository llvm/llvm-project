/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

void
__ockl_atomic_add_noret_f32(float *p, float v)
{
  __opencl_atomic_fetch_add((atomic_float *)p, v, memory_order_relaxed, memory_scope_device);
}

