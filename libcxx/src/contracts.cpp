//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <contracts>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace contracts {

void __default_contract_violation_handler(contract_violation const& __cv) noexcept {
  // TODO: basically reimplement __libcpp_verbose_abort.
}

} // namespace contracts
_LIBCPP_END_NAMESPACE_STD

_LIBCPP_WEAK void handle_contract_violation(std::contracts::contract_violation const& __cv) noexcept
/* strenghtened */ {
  std::contracts::__default_contract_violation_handler(__cv);
}
