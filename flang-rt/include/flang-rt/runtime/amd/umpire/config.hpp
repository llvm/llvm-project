//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_config_HPP
#define UMPIRE_config_HPP

//
// Please keep the list below organized in alphabetical order.
//
/* #undef UMPIRE_ENABLE_BACKTRACE */
/* #undef UMPIRE_ENABLE_BACKTRACE_SYMBOLS */
/* #undef UMPIRE_ENABLE_DEVELOPER_BENCHMARKS */
/* #undef UMPIRE_ENABLE_CONST */
/* #undef UMPIRE_ENABLE_CUDA */
#define UMPIRE_ENABLE_DEVICE
/* #undef UMPIRE_ENABLE_FILESYSTEM */
#define UMPIRE_ENABLE_FILE_RESOURCE
/* #undef UMPIRE_ENABLE_UMAP */
#define UMPIRE_ENABLE_HIP
/* #undef UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY */
/* #undef UMPIRE_ENABLE_IPC_SHARED_MEMORY */
/* #undef UMPIRE_ENABLE_MPI3_SHARED_MEMORY */
/* #undef UMPIRE_ENABLE_INACCESSIBILITY_TESTS */
#define UMPIRE_ENABLE_LOGGING
/* #undef UMPIRE_ENABLE_MPI */
/* #undef UMPIRE_ENABLE_NUMA */
/* #undef UMPIRE_ENABLE_OPENMP_TARGET */
#define UMPIRE_ENABLE_PINNED
/* #undef UMPIRE_ENABLE_SLIC */
/* #undef UMPIRE_ENABLE_SYCL */
#define UMPIRE_ENABLE_UM
/* #undef UMPIRE_ENABLE_ASAN */
/* #undef UMPIRE_ENABLE_DEVICE_ALLOCATOR */
/* #undef UMPIRE_ENABLE_SQLITE_EXPERIMENTAL */
/* #undef UMPIRE_DISABLE_ALLOCATIONMAP_DEBUG */

#define UMPIRE_VERSION_MAJOR 2025
#define UMPIRE_VERSION_MINOR 3
#define UMPIRE_VERSION_PATCH 0
#define UMPIRE_VERSION_RC "6b4cb9e9"

#ifdef __cplusplus

// umpire_EXPORTS gets defined by CMake when we use
// -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=On
#if (defined(_WIN32) || defined(_WIN64)) && !defined(UMPIRE_WIN_STATIC_BUILD)
#ifdef umpire_EXPORTS
#define UMPIRE_EXPORT __declspec(dllexport)
#else
#define UMPIRE_EXPORT __declspec(dllimport)
#endif
#else
#define UMPIRE_EXPORT
#endif

#define UMPIRE_VERSION_SYM  umpire_ver_2025_3_found
UMPIRE_EXPORT extern int    UMPIRE_VERSION_SYM;
#define UMPIRE_VERSION_OK() UMPIRE_VERSION_SYM == 0

namespace umpire {
constexpr int invalid_allocator_id = 0xDEADBEE;
}

#endif

#endif
