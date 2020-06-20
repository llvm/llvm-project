//===-- lib/Parser/tools.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/tools.h"

namespace Fortran::parser {

const Name &GetLastName(const Name &x) { return x; }

const Name &GetLastName(const StructureComponent &x) {
  return GetLastName(x.component);
}

const Name &GetLastName(const DataRef &x) {
  return std::visit(
      common::visitors{
          [](const Name &name) -> const Name & { return name; },
          [](const common::Indirection<StructureComponent> &sc)
              -> const Name & { return GetLastName(sc.value()); },
          [](const common::Indirection<ArrayElement> &sc) -> const Name & {
            return GetLastName(sc.value().base);
          },
          [](const common::Indirection<CoindexedNamedObject> &ci)
              -> const Name & { return GetLastName(ci.value().base); },
      },
      x.u);
}

const Name &GetLastName(const Substring &x) {
  return GetLastName(std::get<DataRef>(x.t));
}

const Name &GetLastName(const Designator &x) {
  return std::visit(
      [](const auto &y) -> const Name & { return GetLastName(y); }, x.u);
}

const Name &GetLastName(const ProcComponentRef &x) {
  return GetLastName(x.v.thing);
}

const Name &GetLastName(const ProcedureDesignator &x) {
  return std::visit(
      [](const auto &y) -> const Name & { return GetLastName(y); }, x.u);
}

const Name &GetLastName(const Call &x) {
  return GetLastName(std::get<ProcedureDesignator>(x.t));
}

const Name &GetLastName(const FunctionReference &x) { return GetLastName(x.v); }

const Name &GetLastName(const Variable &x) {
  return std::visit(
      [](const auto &indirection) -> const Name & {
        return GetLastName(indirection.value());
      },
      x.u);
}

const Name &GetLastName(const AllocateObject &x) {
  return std::visit(
      [](const auto &y) -> const Name & { return GetLastName(y); }, x.u);
}

const Name &GetFirstName(const Name &x) { return x; }

const Name &GetFirstName(const StructureComponent &x) {
  return GetFirstName(x.base);
}

const Name &GetFirstName(const DataRef &x) {
  return std::visit(
      common::visitors{
          [](const Name &name) -> const Name & { return name; },
          [](const common::Indirection<StructureComponent> &sc)
              -> const Name & { return GetFirstName(sc.value()); },
          [](const common::Indirection<ArrayElement> &sc) -> const Name & {
            return GetFirstName(sc.value().base);
          },
          [](const common::Indirection<CoindexedNamedObject> &ci)
              -> const Name & { return GetFirstName(ci.value().base); },
      },
      x.u);
}

const Name &GetFirstName(const Substring &x) {
  return GetFirstName(std::get<DataRef>(x.t));
}

const Name &GetFirstName(const Designator &x) {
  return std::visit(
      [](const auto &y) -> const Name & { return GetFirstName(y); }, x.u);
}

const Name &GetFirstName(const ProcComponentRef &x) {
  return GetFirstName(x.v.thing);
}

const Name &GetFirstName(const ProcedureDesignator &x) {
  return std::visit(
      [](const auto &y) -> const Name & { return GetFirstName(y); }, x.u);
}

const Name &GetFirstName(const Call &x) {
  return GetFirstName(std::get<ProcedureDesignator>(x.t));
}

const Name &GetFirstName(const FunctionReference &x) {
  return GetFirstName(x.v);
}

const Name &GetFirstName(const Variable &x) {
  return std::visit(
      [](const auto &indirect) -> const Name & {
        return GetFirstName(indirect.value());
      },
      x.u);
}

const CoindexedNamedObject *GetCoindexedNamedObject(const DataRef &base) {
  return std::visit(
      common::visitors{
          [](const Name &) -> const CoindexedNamedObject * { return nullptr; },
          [](const common::Indirection<CoindexedNamedObject> &x)
              -> const CoindexedNamedObject * { return &x.value(); },
          [](const auto &x) -> const CoindexedNamedObject * {
            return GetCoindexedNamedObject(x.value().base);
          },
      },
      base.u);
}
const CoindexedNamedObject *GetCoindexedNamedObject(
    const AllocateObject &allocateObject) {
  return std::visit(
      common::visitors{
          [](const StructureComponent &x) -> const CoindexedNamedObject * {
            return GetCoindexedNamedObject(x.base);
          },
          [](const auto &) -> const CoindexedNamedObject * { return nullptr; },
      },
      allocateObject.u);
}
} // namespace Fortran::parser
