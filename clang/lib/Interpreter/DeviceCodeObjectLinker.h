//===--- DeviceCodeObjectLinker.h - Device Code Object Linker ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_LIB_INTERPRETER_DEVICECODEOBJECTLINKER_H
#define LLVM_CLANG_LIB_INTERPRETER_DEVICECODEOBJECTLINKER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>

namespace clang {

// Turns one relocatable device object into a loadable code object.
class DeviceCodeObjectLinker {
public:
  virtual ~DeviceCodeObjectLinker() = default;
  virtual llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  link(llvm::MemoryBufferRef Object) = 0;
};

llvm::Expected<std::unique_ptr<DeviceCodeObjectLinker>>
selectDeviceCodeObjectLinker(llvm::ArrayRef<const char *> DriverArgs);

} // namespace clang

#endif
