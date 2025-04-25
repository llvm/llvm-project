//===- comgr-device-libs.h - Handle AMD Device Libraries ------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_DEVICE_LIBS_H
#define COMGR_DEVICE_LIBS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <tuple>

namespace COMGR {

struct DataAction;
struct DataSet;

llvm::ArrayRef<unsigned char> getDeviceLibrariesIdentifier();
llvm::StringRef getOpenCLCBaseHeaderContents();
llvm::ArrayRef<std::tuple<llvm::StringRef, llvm::StringRef>>
getDeviceLibraries();

} // namespace COMGR

#endif // COMGR_DEVICE_LIBS_H
