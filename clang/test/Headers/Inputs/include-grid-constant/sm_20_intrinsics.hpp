// Minimal stand-in for the CUDA header's __isGridConstant definition. The
// wrapper renames this function before including the header; without the
// rename, the wrapper's own definition would be a redefinition.
#pragma once

static __device__ __forceinline__ unsigned int __isGridConstant(const void *p) {
  return 0;
}
