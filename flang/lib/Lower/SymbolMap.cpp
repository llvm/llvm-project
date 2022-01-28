//===-- SymbolMap.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pretty printers for symbol boxes, etc.
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/SymbolMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-lower-symbol-map"

void Fortran::lower::SymMap::addSymbol(Fortran::semantics::SymbolRef sym,
                                       const fir::ExtendedValue &exv,
                                       bool force) {
  exv.match([&](const fir::UnboxedValue &v) { addSymbol(sym, v, force); },
            [&](const fir::CharBoxValue &v) { makeSym(sym, v, force); },
            [&](const fir::ArrayBoxValue &v) { makeSym(sym, v, force); },
            [&](const fir::CharArrayBoxValue &v) { makeSym(sym, v, force); },
            [&](const fir::BoxValue &v) { makeSym(sym, v, force); },
            [&](const fir::MutableBoxValue &v) { makeSym(sym, v, force); },
            [](auto) {
              llvm::report_fatal_error("value not added to symbol table");
            });
}

Fortran::lower::SymbolBox
Fortran::lower::SymMap::lookupSymbol(Fortran::semantics::SymbolRef sym) {
  for (auto jmap = symbolMapStack.rbegin(), jend = symbolMapStack.rend();
       jmap != jend; ++jmap) {
    auto iter = jmap->find(&*sym);
    if (iter != jmap->end())
      return iter->second;
  }
  return SymbolBox::None{};
}

mlir::Value
Fortran::lower::SymMap::lookupImpliedDo(Fortran::lower::SymMap::AcDoVar var) {
  for (auto [marker, binding] : llvm::reverse(impliedDoStack))
    if (var == marker)
      return binding;
  return {};
}

llvm::raw_ostream &
Fortran::lower::operator<<(llvm::raw_ostream &os,
                           const Fortran::lower::SymbolBox &symBox) {
  symBox.match(
      [&](const Fortran::lower::SymbolBox::None &box) {
        os << "** symbol not properly mapped **\n";
      },
      [&](const Fortran::lower::SymbolBox::Intrinsic &val) {
        os << val.getAddr() << '\n';
      },
      [&](const auto &box) { os << box << '\n'; });
  return os;
}

llvm::raw_ostream &
Fortran::lower::operator<<(llvm::raw_ostream &os,
                           const Fortran::lower::SymMap &symMap) {
  os << "Symbol map:\n";
  for (auto i : llvm::enumerate(symMap.symbolMapStack)) {
    os << " level " << i.index() << "<{\n";
    for (auto iter : i.value())
      os << "  symbol @" << static_cast<const void *>(iter.first) << " ["
         << *iter.first << "] ->\n    " << iter.second;
    os << " }>\n";
  }
  return os;
}
