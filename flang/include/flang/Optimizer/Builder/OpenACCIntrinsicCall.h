//===-- Builder/OpenACCIntrinsicCall.h - OpenACC intrinsics -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_OPENACCINTRINSICCALL_H
#define FORTRAN_LOWER_OPENACCINTRINSICCALL_H

#include "flang/Optimizer/Builder/IntrinsicCall.h"

namespace fir {

struct OpenACCIntrinsicLibrary : IntrinsicLibrary {

  explicit OpenACCIntrinsicLibrary(fir::FirOpBuilder &builder,
                                   mlir::Location loc)
      : IntrinsicLibrary(builder, loc) {}
  OpenACCIntrinsicLibrary() = delete;
  OpenACCIntrinsicLibrary(const OpenACCIntrinsicLibrary &) = delete;

  mlir::Value genACCOnDevice(mlir::Type, llvm::ArrayRef<mlir::Value>);
};

const IntrinsicHandler *findOpenACCIntrinsicHandler(llvm::StringRef name,
                                                    bool isBindcCall = false);

} // namespace fir

#endif // FORTRAN_LOWER_OPENACCINTRINSICCALL_H
