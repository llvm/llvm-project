//===-- runtime/derived.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RUNTIME_DERIVED_H_
#define FLANG_RUNTIME_DERIVED_H_

namespace Fortran::runtime::typeInfo {
class DerivedType;
}

namespace Fortran::runtime {
class Descriptor;
void Destroy(const Descriptor &, bool finalize, const typeInfo::DerivedType &);
} // namespace Fortran::runtime
#endif // FLANG_RUNTIME_FINAL_H_
