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
__attribute__((always_inline)) extern uint8_t
hvx_quantize_f32_to_u8(float scale_f, int16_t out_offset, const float in);
__attribute__((always_inline)) extern uint16_t
hvx_quantize_f32_to_u16(float scale_f, int32_t out_offset, const float in);
#ifdef __cplusplus
}
#endif
