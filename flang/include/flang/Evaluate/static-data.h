//===-- include/flang/Evaluate/static-data.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_STATIC_DATA_H_
#define FORTRAN_EVALUATE_STATIC_DATA_H_

// Represents constant static data objects

#include "formatting.h"
#include "type.h"
#include "flang/Common/idioms.h"
#include <cinttypes>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace Fortran::evaluate {

class StaticDataObject {
public:
  using Pointer = std::shared_ptr<StaticDataObject>;

  StaticDataObject(const StaticDataObject &) = delete;
  StaticDataObject(StaticDataObject &&) = delete;
  StaticDataObject &operator=(const StaticDataObject &) = delete;
  StaticDataObject &operator=(StaticDataObject &&) = delete;

  static Pointer Create() { return Pointer{new StaticDataObject}; }

  const std::string &name() const { return name_; }
  StaticDataObject &set_name(std::string n) {
    name_ = n;
    return *this;
  }

  int alignment() const { return alignment_; }
  StaticDataObject &set_alignment(int a) {
    CHECK(a >= 0);
    alignment_ = a;
    return *this;
  }

  int itemBytes() const { return itemBytes_; }
  StaticDataObject &set_itemBytes(int b) {
    CHECK(b >= 1);
    itemBytes_ = b;
    return *this;
  }

  const std::vector<std::uint8_t> &data() const { return data_; }
  std::vector<std::uint8_t> &data() { return data_; }

  StaticDataObject &Push(const std::string &);
  StaticDataObject &Push(const std::u16string &);
  StaticDataObject &Push(const std::u32string &);
  std::optional<std::string> AsString() const;
  std::optional<std::u16string> AsU16String() const;
  std::optional<std::u32string> AsU32String() const;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;

  static bool bigEndian;

private:
  StaticDataObject() {}

  std::string name_;
  int alignment_{1};
  int itemBytes_{1};
  std::vector<std::uint8_t> data_;
};
} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_STATIC_DATA_H_
