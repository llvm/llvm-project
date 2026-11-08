//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Storage-independent access to the register values of an execution context.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_REGISTERCONTEXT_H
#define LLDB_REGISTER_REGISTERCONTEXT_H

#include "lldb/Register/RegisterCheckpoint.h"
#include "lldb/Register/RegisterDescription.h"
#include "lldb/Register/RegisterValue.h"

namespace lldb_private {

/// Access to logical register values without prescribing a saved-state format.
///
/// Implementations acquire state from a process, remote stub, or saved image
/// and provide an immutable RegisterDescription. This interface resolves
/// metadata and numbering; acquisition, cache policy, frame selection, and
/// native APIs remain in the implementation. Access by RegisterID must not be
/// confused with enumeration by a description or user-visible table index.
class RegisterContext {
public:
  /// Use a shared backend owner identity for checkpoints, or create an identity
  /// unique to this adapter when no owner is supplied. The external owner is
  /// observed weakly and must remain alive for checkpoint restoration.
  explicit RegisterContext(std::shared_ptr<const void> owner = {});
  RegisterContext(const RegisterContext &) = delete;
  RegisterContext &operator=(const RegisterContext &) = delete;
  virtual ~RegisterContext() = default;

  /// Return the current immutable metadata, or nullptr while it is unavailable.
  /// If the register configuration changes, publish a replacement description.
  virtual std::shared_ptr<const RegisterDescription> GetDescription() const = 0;
  size_t GetRegisterCount() const;
  size_t GetUserRegisterCount() const;

  /// Look up a stored table index; return nullptr when metadata or index is
  /// absent.
  const RegisterDescriptor *GetRegisterAtIndex(size_t index) const;

  /// Look up a user presentation index; return nullptr when it is out of range.
  const RegisterDescriptor *GetUserRegisterAtIndex(size_t index) const;
  size_t GetRegisterGroupCount() const;

  /// Return a presentation group, or nullptr when metadata or index is absent.
  const RegisterGroup *GetRegisterGroup(size_t index) const;

  /// Resolve debugger names, including generic roles and alternate names.
  /// Return nullptr when no matching register is available.
  const RegisterDescriptor *
  GetRegisterByName(llvm::StringRef name, uint32_t start_index = 0) const;

  /// Resolve a number in the requested scheme, or return nullptr if unmapped.
  const RegisterDescriptor *GetRegisterByNumber(uint32_t kind,
                                                uint32_t number) const;
  const char *GetRegisterNameAtIndex(uint32_t index) const;

  /// Return the logical identity for an external number, or
  /// LLDB_INVALID_REGNUM.
  RegisterID GetRegisterIDForNumber(uint32_t kind, uint32_t number) const;

  /// Convert numbering schemes through the current description. Return false
  /// when either scheme or the register mapping is unavailable.
  bool ConvertBetweenRegisterKinds(lldb::RegisterKind source, uint32_t number,
                                   lldb::RegisterKind target,
                                   uint32_t &result) const;

  /// Most sources resolve numbers from their immutable description. Virtual
  /// machines may also resolve an encoded, lazily described value identity.
  virtual llvm::Expected<RegisterID>
  ResolveRegisterNumber(lldb::RegisterKind kind, uint32_t number) const;

  /// Read a logical value, reporting unknown, unavailable, or inaccessible
  /// state as an error. The returned value must own any bytes acquired by the
  /// backend.
  virtual llvm::Expected<RegisterValue> ReadRegister(RegisterID id) = 0;

  /// Write a logical value and report backend or encoding failures.
  /// Implementations are responsible for invalidating dependent cached values.
  virtual llvm::Error WriteRegister(RegisterID id,
                                    const RegisterValue &value) = 0;

  /// Read and convert to an unsigned scalar, returning the supplied fallback
  /// when reading or conversion fails.
  uint64_t ReadRegisterAsUnsigned(RegisterID id, uint64_t fail_value);

  /// Encode an unsigned scalar at the described width, then delegate the write.
  /// Unknown IDs, unsupported widths, and backend failures are errors.
  llvm::Error WriteRegisterFromUnsigned(RegisterID id, uint64_t value);

  /// Discard cached state so subsequent reads reacquire it. The default
  /// implementation has no cache to invalidate.
  virtual void InvalidateAllRegisters() {}

  /// Capture an opaque backend restore operation, or return an error when
  /// checkpointing is unsupported. Backend-specific buffers, remote tokens, and
  /// configuration restoration remain inside the returned operation.
  virtual llvm::Expected<RegisterCheckpoint> SaveCheckpoint();

  /// Restore a checkpoint only while its captured owner is alive and matches
  /// this context. Return an owner error before invoking a foreign or expired
  /// operation; otherwise propagate the backend restoration result.
  llvm::Error RestoreCheckpoint(const RegisterCheckpoint &checkpoint);

protected:
  /// If an external owner can outlive this adapter, retain the adapter in the
  /// restore operation. Uniquely owned adapters use the default owner identity.
  RegisterCheckpoint
  CreateCheckpoint(std::function<llvm::Error()> restore) const;

private:
  std::shared_ptr<const void> m_owned_identity;
  std::weak_ptr<const void> m_checkpoint_owner;
};

} // namespace lldb_private
#endif
