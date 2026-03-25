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

/// Plugin interface for building structured types to represent CPU registers.
///
/// RegisterTypeBuilder plugins create CompilerType representations for CPU
/// registers that have associated register fields (flags). These structured
/// types allow debuggers to display register contents in a human-readable
/// format, showing individual bit fields and their meanings rather than just
/// raw hexadecimal values.
///
/// LLDB uses these plugins when displaying registers that have associated
/// RegisterFlags metadata. When a register with flags needs to be displayed,
/// LLDB queries the active RegisterTypeBuilder plugin (if any) to create a
/// structured type that represents the register's bit fields. This type is
/// then used by the type system to format and display the register value.
///
/// Plugin Selection and Instantiation:
/// RegisterTypeBuilder plugins are instantiated per-target and are selected
/// based on the target's type system. For example, RegisterTypeBuilderClang
/// is used for targets that use Clang as their primary type system. The plugin
/// is created via the CreateInstance callback registered with the PluginManager.
///
/// Implementation Considerations:
/// - Implementations should cache created types when possible, as the same
///   register type may be requested multiple times
/// - The returned CompilerType must accurately reflect the register's bit
///   layout, including field positions, sizes, and types
/// - Field names should match the flag names defined in RegisterFlags metadata
/// - Consider endianness when constructing the type representation
class RegisterTypeBuilder : public PluginInterface {
public:
  ~RegisterTypeBuilder() override = default;

  virtual CompilerType GetRegisterType(const std::string &name,
                                       const lldb_private::RegisterFlags &flags,
                                       uint32_t byte_size) = 0;

protected:
  RegisterTypeBuilder() = default;

private:
  RegisterTypeBuilder(const RegisterTypeBuilder &) = delete;
  const RegisterTypeBuilder &operator=(const RegisterTypeBuilder &) = delete;
};

} // namespace lldb_private

#endif // LLDB_TARGET_REGISTERTYPEBUILDER_H
