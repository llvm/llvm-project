//===-- copyprof_interface_internal.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the internal runtime interface for CopyProf, including
/// initialization and special member function callback declarations.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_INTERFACE_INTERNAL_H
#define COPYPROF_INTERFACE_INTERNAL_H

#include "copyprof_internal.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

extern "C" {

// Should be called at the very beginning of the process before any instrumented
// code executes.
SANITIZER_INTERFACE_ATTRIBUTE void __copyprof_init();

// Runtime callbacks that update the CopyProf state machine and shadow memory
// when entering or leaving special member functions.
SANITIZER_INTERFACE_ATTRIBUTE void __copyprof_ctor_enter_callback(
    const void* this_ptr, uptr obj_size);
SANITIZER_INTERFACE_ATTRIBUTE void __copyprof_ctor_exit_callback(
    const void* this_ptr, uptr obj_size);
SANITIZER_INTERFACE_ATTRIBUTE void __copyprof_copy_ctor_enter_callback(
    const void* this_ptr, const void* other_ptr, uptr obj_size);
SANITIZER_INTERFACE_ATTRIBUTE void __copyprof_copy_ctor_exit_callback(
    const void* this_ptr, const void* other_ptr, uptr obj_size);
SANITIZER_INTERFACE_ATTRIBUTE void __copyprof_copy_assign_op_enter_callback(
    const void* this_ptr, const void* other_ptr, uptr obj_size);
SANITIZER_INTERFACE_ATTRIBUTE void __copyprof_copy_assign_op_exit_callback(
    const void* this_ptr, const void* other_ptr, uptr obj_size);
SANITIZER_INTERFACE_ATTRIBUTE void __copyprof_dtor_enter_callback(
    const void* this_ptr, uptr obj_size);
SANITIZER_INTERFACE_ATTRIBUTE void __copyprof_dtor_exit_callback(
    const void* this_ptr, uptr obj_size);
// Store instructions callback that marks memory as not copy.
SANITIZER_INTERFACE_ATTRIBUTE void __copyprof_store_callback(const void* addr,
                                                             uptr size);

}  // extern "C"

#endif  // COPYPROF_INTERFACE_INTERNAL_H
