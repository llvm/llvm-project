/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#pragma once
#include "asan_util.h"

// The strucutures semantics and layout must match the host instrumented
// global variable as defined in
// llvm-project/compiler-rt/lib/asan/asan_interface_internal.h

// This structure used to describe the source location of a place
// where global was defined.
struct global_source_location {
    const char *filename;
    int line_no;
    int column_no;
};

// This structure describes an instrumented global variable.
struct device_global {
    uptr beg;                // The address of the global.
    uptr size;               // The original size of the global.
    uptr size_with_redzone;  // The size with the redzone.
    const char *name;        // Name as a C string.
    const char *module_name; // Module name as a C string. This pointer is a
                             // unique identifier of a module.
    uptr has_dynamic_init;   // Non-zero if the global has dynamic initializer.
    struct global_source_location *location; // Source location of a global,
                                             // or NULL if it is unknown.
    uptr odr_indicator; // The address of the ODR indicator symbol.
};

static const __constant s8 kAsanGlobalRedzoneMagic = 0xf9;
