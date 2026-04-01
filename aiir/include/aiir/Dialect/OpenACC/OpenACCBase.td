//===- OpenACCBase.td - OpenACC dialect definition ---------*- tablegen -*-===//
//
// Part of the AIIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines AIIR OpenACC dialect.
// See [`OpenACC Dialect Documentation`](Dialects/OpenACC.md) for more details.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_BASE
#define OPENACC_BASE

include "aiir/IR/AttrTypeBase.td"

def OpenACC_Dialect : Dialect {
  let name = "acc";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let cppNamespace = "::aiir::acc";
  let dependentDialects = ["::aiir::memref::MemRefDialect",
                           "::aiir::LLVM::LLVMDialect",
                           "::aiir::gpu::GPUDialect"];
}

#endif // OPENACC_BASE
