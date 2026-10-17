//===-- Semantics/openacc-cuda-matching.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OPENACC_CUDA_MATCHING_H_
#define FORTRAN_SEMANTICS_OPENACC_CUDA_MATCHING_H_

#include <cstddef>
#include <limits>

namespace Fortran::semantics {

// These values index the table below: dummy selects a row, and
// 3 * storage + mode selects a column. Keep them aligned with its layout.
enum class CUDAMemoryMode { Default = 0, Managed = 1, Unified = 2 };
enum class OpenACCActualStorage { Dynamic = 0, Fixed = 1 };
enum class CUDADummyCategory { Host = 0, Device = 1, Managed = 2, Unified = 3 };

inline constexpr int CUDAInfMatchingValue{std::numeric_limits<int>::max()};

// Scores an unattributed variable passed by reference from an unconditional
// OpenACC GPU compute body. Explicit CUDA attributes and VALUE arguments use
// ordinary CUDA matching instead. Dynamic includes allocatable/pointer actuals;
// private and fixed-size objects use Fixed. Lower scores are preferred, and INF
// rejects the candidate.
constexpr int GetOpenACCGPUMatchingDistance(CUDAMemoryMode mode,
    OpenACCActualStorage storage, CUDADummyCategory dummy) {
  constexpr int INF{CUDAInfMatchingValue};
  //                            Dynamic               Fixed/private
  //                       Default Managed Unified Default Managed Unified
  constexpr int distance[][6]{
      /* Host */ {1, 3, 2, 1, 1, 2},
      /* Device */ {0, 0, 1, 0, 0, 1},
      /* Managed */ {INF, 1, 0, INF, INF, 0},
      /* Unified */ {INF, 2, 3, INF, INF, 3},
  };
  return distance[static_cast<std::size_t>(dummy)]
                 [3 * static_cast<std::size_t>(storage) +
                     static_cast<std::size_t>(mode)];
}

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_OPENACC_CUDA_MATCHING_H_
