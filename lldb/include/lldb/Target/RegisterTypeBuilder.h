//===-- RegisterTypeBuilder.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_REGISTERTYPEBUILDER_H
#define LLDB_TARGET_REGISTERTYPEBUILDER_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class RegisterUnion;

class RegisterTypeBuilder : public PluginInterface {
public:
  ~RegisterTypeBuilder() override = default;

  virtual CompilerType GetRegisterType(const std::string &name,
                                       const lldb_private::RegisterFlags &flags,
                                       uint32_t byte_size) = 0;

  virtual CompilerType
  GetRegisterUnionType(const std::string &name,
                       const lldb_private::RegisterUnion &union_type,
                       uint32_t byte_size) = 0;

protected:
  RegisterTypeBuilder() = default;

private:
  RegisterTypeBuilder(const RegisterTypeBuilder &) = delete;
  const RegisterTypeBuilder &operator=(const RegisterTypeBuilder &) = delete;
};

} // namespace lldb_private

#endif // LLDB_TARGET_REGISTERTYPEBUILDER_H
