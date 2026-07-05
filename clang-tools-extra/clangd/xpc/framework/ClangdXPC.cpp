//===-- ClangdXPC.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// Returns the bundle identifier of the Clangd XPC service.
extern "C" const char *clangd_xpc_get_bundle_identifier() {
  return "org.llvm.clangd";
}
