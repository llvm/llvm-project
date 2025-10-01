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
#include "flang/Optimizer/Builder/Todo.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"
#include <optional>

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
            [&](const fir::PolymorphicValue &v) { makeSym(sym, v, force); },
            [](auto) {
              llvm::report_fatal_error("value not added to symbol table");
            });
}

Fortran::lower::SymbolBox
Fortran::lower::SymMap::lookupSymbol(Fortran::semantics::SymbolRef symRef) {
  auto *sym = symRef->HasLocalLocality() ? &*symRef : &symRef->GetUltimate();
  for (auto jmap = symbolMapStack.rbegin(), jend = symbolMapStack.rend();
       jmap != jend; ++jmap) {
    auto iter = jmap->find(sym);
    if (iter != jmap->end())
      return iter->second;
  }
  return SymbolBox::None{};
}

Fortran::lower::SymbolBox Fortran::lower::SymMap::shallowLookupSymbol(
    Fortran::semantics::SymbolRef symRef) {
  auto *sym = symRef->HasLocalLocality() ? &*symRef : &symRef->GetUltimate();
  auto &map = symbolMapStack.back();
  auto iter = map.find(sym);
  if (iter != map.end())
    return iter->second;
  return SymbolBox::None{};
}

/// Skip one level when looking up the symbol. The use case is such as looking
/// up the host variable symbol box by skipping the associated level in
/// host-association in OpenMP code.
Fortran::lower::SymbolBox Fortran::lower::SymMap::lookupOneLevelUpSymbol(
    Fortran::semantics::SymbolRef symRef) {
  auto *sym = symRef->HasLocalLocality() ? &*symRef : &symRef->GetUltimate();
  auto jmap = symbolMapStack.rbegin();
  auto jend = symbolMapStack.rend();
  if (jmap == jend)
    return SymbolBox::None{};
  // Skip one level in symbol map stack.
  for (++jmap; jmap != jend; ++jmap) {
    auto iter = jmap->find(sym);
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

void Fortran::lower::SymMap::registerStorage(
    semantics::SymbolRef symRef, Fortran::lower::SymMap::StorageDesc storage) {
  auto *sym = symRef->HasLocalLocality() ? &*symRef : &symRef->GetUltimate();
  assert(storage.first && "registerting storage without an address");
  storageMapStack.back().insert_or_assign(sym, std::move(storage));
}

Fortran::lower::SymMap::StorageDesc
Fortran::lower::SymMap::lookupStorage(Fortran::semantics::SymbolRef symRef) {
  auto *sym = symRef->HasLocalLocality() ? &*symRef : &symRef->GetUltimate();
  auto &map = storageMapStack.back();
  auto iter = map.find(sym);
  if (iter != map.end())
    return iter->second;
  return {nullptr, 0};
}

void Fortran::lower::SymbolBox::dump() const { llvm::errs() << *this << '\n'; }

void Fortran::lower::SymMap::dump() const { llvm::errs() << *this << '\n'; }

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
    for (auto iter : i.value()) {
      os << "  symbol @" << static_cast<const void *>(iter.first) << " ["
         << *iter.first << "] ->\n    ";
      os << iter.second;
    }
    os << " }>\n";
  }
  return os;
}
