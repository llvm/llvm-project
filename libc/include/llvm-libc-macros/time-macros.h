#ifndef LLVM_LIBC_MACROS_TIME_MACROS_H
#define LLVM_LIBC_MACROS_TIME_MACROS_H

#if defined(__AMDGPU__) || defined(__NVPTX__)
#include "gpu/time-macros.h"
#elif defined(__linux__)
#include "linux/time-macros.h"
#endif

#endif // LLVM_LIBC_MACROS_TIME_MACROS_H
