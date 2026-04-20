//==============================================================================
//
// Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==============================================================================
//

#pragma once
#include "__ripple_vec.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
__attribute__((always_inline)) extern uint32_t
hvx_dequantize_u8_to_f32(const uint8_t vin, const int16_t zero_offset,
                         const uint8_t from_qint8, const int32_t iscale,
                         const int16_t exp_base);
__attribute__((always_inline)) extern float
hvx_dequantize_u16_to_f32_flat(const uint16_t vin, uint32_t offset,
                               float scale);
__attribute__((always_inline)) extern uint32_t
hvx_dequantize_u16_to_f32(const uint16_t vin, uint16_t zero_offset,
                          uint16_t from_qint16, int32_t iscale,
                          int16_t exp_base);
#ifdef __cplusplus
}
#endif
