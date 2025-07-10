//===-- RemoteJITUtils.h - Utilities for remote-JITing ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for ExecutorProcessControl-based remote JITing with Orc and
// JITLink.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INTERPRETER_REMOTEJITUTILS_H
#define LLVM_CLANG_INTERPRETER_REMOTEJITUTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>

llvm::Expected<std::unique_ptr<llvm::orc::SimpleRemoteEPC>>
launchExecutor(llvm::StringRef ExecutablePath, bool UseSharedMemory,
               llvm::StringRef SlabAllocateSizeString,
               std::function<void()> CustomizeFork = nullptr);

/// Create a JITLinkExecutor that connects to the given network address
/// through a TCP socket. A valid NetworkAddress provides hostname and port,
/// e.g. localhost:20000.
llvm::Expected<std::unique_ptr<llvm::orc::SimpleRemoteEPC>>
connectTCPSocket(llvm::StringRef NetworkAddress, bool UseSharedMemory,
                 llvm::StringRef SlabAllocateSizeString);

#ifdef LLVM_ON_UNIX
/// Returns PID of last launched executor.
pid_t getLastLaunchedExecutorPID();

/// Returns PID of nth launched executor.
/// 1-based indexing.
pid_t getNthLaunchedExecutorPID(int n);
#endif

#endif // LLVM_CLANG_INTERPRETER_REMOTEJITUTILS_H
