#ifndef __CLC_INTERNAL_CLC_H_
#define __CLC_INTERNAL_CLC_H_

#ifndef cl_clang_storage_class_specifiers
#error Implementation requires cl_clang_storage_class_specifiers extension!
#endif

#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

/* Function Attributes */
#include <clc/clcfunc.h>

/* 6.1 Supported Data Types */
#include <clc/clctypes.h>

/* 6.2.4.2 Reinterpreting Types Using __clc_as_type() and __clc_as_typen() */
#include <clc/clc_as_type.h>

#pragma OPENCL EXTENSION all : disable

#endif // __CLC_INTERNAL_CLC_H_
