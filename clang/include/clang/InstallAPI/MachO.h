//===- InstallAPI/MachO.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Imports and forward declarations for llvm::MachO types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INSTALLAPI_MACHO_H
#define LLVM_CLANG_INSTALLAPI_MACHO_H

#include "llvm/TextAPI/Architecture.h"
#include "llvm/TextAPI/InterfaceFile.h"
#include "llvm/TextAPI/PackedVersion.h"
#include "llvm/TextAPI/Platform.h"
#include "llvm/TextAPI/RecordVisitor.h"
#include "llvm/TextAPI/Target.h"
#include "llvm/TextAPI/TextAPIWriter.h"
#include "llvm/TextAPI/Utils.h"

using SymbolFlags = llvm::MachO::SymbolFlags;
using RecordLinkage = llvm::MachO::RecordLinkage;
using Record = llvm::MachO::Record;
using GlobalRecord = llvm::MachO::GlobalRecord;
using ObjCContainerRecord = llvm::MachO::ObjCContainerRecord;
using ObjCInterfaceRecord = llvm::MachO::ObjCInterfaceRecord;
using ObjCCategoryRecord = llvm::MachO::ObjCCategoryRecord;
using ObjCIVarRecord = llvm::MachO::ObjCIVarRecord;
using Records = llvm::MachO::Records;
using BinaryAttrs = llvm::MachO::RecordsSlice::BinaryAttrs;
using SymbolSet = llvm::MachO::SymbolSet;
using FileType = llvm::MachO::FileType;
using PackedVersion = llvm::MachO::PackedVersion;
using Target = llvm::MachO::Target;

#endif // LLVM_CLANG_INSTALLAPI_MACHO_H
