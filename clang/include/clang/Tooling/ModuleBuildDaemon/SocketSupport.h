//===-------------------------- SocketSupport.h ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETSUPPORT_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETSUPPORT_H

#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

using namespace clang;
using namespace llvm;

namespace cc1modbuildd {

Expected<int> createSocket();
Expected<int> connectToSocket(StringRef SocketPath);
Expected<int> connectAndWriteToSocket(std::string Buffer, StringRef SocketPath);
Expected<std::string> readFromSocket(int FD);
llvm::Error writeToSocket(std::string Buffer, int WriteFD);

} // namespace cc1modbuildd

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETSUPPORT_H
