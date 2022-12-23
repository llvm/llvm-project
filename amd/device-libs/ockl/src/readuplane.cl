/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ockl.h"

#define WAVESIZE 64


// Function to exchange data between different lanes
// var: value to return if the index is outside the bounds of the wave
// offset: To be added to the lane id to obtain final index
// return a int value correspoding to the lane

int
__ockl_readuplane_i32(int var, int offset)
{

     uint lane_id = __ockl_lane_u32();
     int index = lane_id + offset;
     index = (uint)((lane_id & (WAVESIZE - 1)) + offset) >= WAVESIZE ? lane_id : index;
     return __builtin_amdgcn_ds_bpermute(index << 2, var);
 }


// Function to exchange data between different lanes
// var: value to return if the index is outside the bounds of the wave
// offset: To be added to the lane id to obtain final index
// return a long value correspoding to the lane

long
__ockl_readuplane_i64(long var, int offset) {
  int lane_id = __ockl_lane_u32();
  int index = lane_id + offset;
  index = (uint)((lane_id & (WAVESIZE - 1)) + offset) >= WAVESIZE ? lane_id : index;
  int2 var_64= __builtin_astype(var, int2);
  var_64.x =  __builtin_amdgcn_ds_bpermute(index << 2, var_64.x);
  var_64.y =  __builtin_amdgcn_ds_bpermute(index << 2, var_64.y);
  return __builtin_astype(var_64, long);
}
