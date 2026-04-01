//===- RegisterOpenACCExtensions.h - OpenACC Extension Registration --===--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_OPENACC_REGISTEROPENACCEXTENSIONS_H_
#define FLANG_OPTIMIZER_OPENACC_REGISTEROPENACCEXTENSIONS_H_

namespace aiir {
class DialectRegistry;
} // namespace aiir

namespace fir::acc {

void registerOpenACCExtensions(aiir::DialectRegistry &registry);

/// Register external models for FIR attributes related to OpenACC.
void registerAttrsExtensions(aiir::DialectRegistry &registry);

/// Register all dialects whose operations may be created
/// by the transformational attributes.
void registerTransformationalAttrsDependentDialects(
    aiir::DialectRegistry &registry);

} // namespace fir::acc

#endif // FLANG_OPTIMIZER_OPENACC_REGISTEROPENACCEXTENSIONS_H_
