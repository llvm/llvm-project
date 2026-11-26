//===-- OpenACCIntrinsicCall.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/OpenACCIntrinsicCall.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"

namespace fir {

using OAI = OpenACCIntrinsicLibrary;

static constexpr IntrinsicHandler openaccHandlers[]{
    {"acc_on_device",
     static_cast<OpenACCIntrinsicLibrary::ElementalGenerator>(
         &OAI::genACCOnDevice),
     {{{"devtype", asValue}}},
     /*isElemental=*/false},
};
static_assert(fir::isSorted(openaccHandlers) && "map must be sorted");

const IntrinsicHandler *findOpenACCIntrinsicHandler(llvm::StringRef name,
                                                    bool isBindcCall) {
  if (!isBindcCall)
    return nullptr;
  auto compare = [](const IntrinsicHandler &openaccHandler,
                    llvm::StringRef name) {
    return name.compare(openaccHandler.name) > 0;
  };
  auto result = llvm::lower_bound(openaccHandlers, name, compare);
  return result != std::end(openaccHandlers) && result->name == name ? result
                                                                     : nullptr;
}

mlir::Value
OpenACCIntrinsicLibrary::genACCOnDevice(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  mlir::Value onDevice =
      mlir::acc::OnDeviceOp::create(builder, loc, args[0]).getResult();
  return builder.createConvert(loc, resultType, onDevice);
}

} // namespace fir
