//===- Writer.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_WRITER_H
#define LLD_MACHO_WRITER_H

namespace lld {
namespace macho {

void writeResult();

void createSyntheticSections();

} // namespace macho
} // namespace lld

#endif
