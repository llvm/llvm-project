//===-- At-fork callback helpers  -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace __llvm_libc {

using ForkCallback = void(void);

bool register_atfork_callbacks(ForkCallback *prepare_cd,
                               ForkCallback *parent_cb, ForkCallback *child_cb);
void invoke_prepare_callbacks();
void invoke_parent_callbacks();
void invoke_child_callbacks();

} // namespace __llvm_libc
