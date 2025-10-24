//===-- SyntheticFrameProvider.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_SYNTHETICFRAMEPROVIDER_H
#define LLDB_TARGET_SYNTHETICFRAMEPROVIDER_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <vector>

namespace lldb_private {

/// Descriptor for configuring a synthetic frame provider.
///
/// This struct contains the metadata needed to instantiate a frame provider
/// and optional filters to control which threads it applies to.
struct SyntheticFrameProviderDescriptor {
  /// Metadata for instantiating the provider (e.g., script class name and args)
  lldb::ScriptedMetadataSP scripted_metadata_sp;

  /// Optional list of thread IDs to which this provider applies.
  /// If empty, the provider applies to all threads.
  std::vector<lldb::tid_t> thread_ids;

  SyntheticFrameProviderDescriptor() = default;

  SyntheticFrameProviderDescriptor(lldb::ScriptedMetadataSP metadata_sp)
      : scripted_metadata_sp(metadata_sp) {}

  SyntheticFrameProviderDescriptor(lldb::ScriptedMetadataSP metadata_sp,
                                   const std::vector<lldb::tid_t> &tids)
      : scripted_metadata_sp(metadata_sp), thread_ids(tids) {}

  /// Check if this descriptor applies to the given thread ID.
  bool AppliesToThread(lldb::tid_t tid) const {
    // If no thread IDs specified, applies to all threads
    if (thread_ids.empty())
      return true;

    // Check if the thread ID is in the filter list
    return std::find(thread_ids.begin(), thread_ids.end(), tid) !=
           thread_ids.end();
  }

  /// Check if this descriptor has valid metadata.
  bool IsValid() const { return scripted_metadata_sp != nullptr; }
};

/// Base class for all synthetic frame providers.
///
/// Synthetic frame providers allow modifying or replacing the stack frames
/// shown for a thread. This is useful for:
/// - Providing frames for custom calling conventions or languages
/// - Reconstructing missing frames from crash dumps or core files
/// - Adding diagnostic or synthetic frames for debugging
/// - Visualizing state machines or async execution contexts
class SyntheticFrameProvider : public PluginInterface {
public:
  /// Try to create a SyntheticFrameProvider instance for the given thread
  /// and metadata.
  ///
  /// This method iterates through all registered SyntheticFrameProvider
  /// plugins and returns the first one that can handle the given metadata.
  ///
  /// \param[in] thread_sp
  ///     The thread for which to provide synthetic frames.
  ///
  /// \return
  ///     A shared pointer to a SyntheticFrameProvider if one could be created,
  ///     otherwise an \a llvm::Error.
  static llvm::Expected<lldb::SyntheticFrameProviderSP>
  CreateInstance(lldb::ThreadSP thread_sp);

  ~SyntheticFrameProvider() override;

  /// Get the stack frames from the synthetic frame provider.
  ///
  /// \param[in] real_frames
  ///     The actual unwound frames from the thread's normal unwinder.
  ///
  /// \return
  ///     An Expected containing the StackFrameListSP if successful,
  ///     otherwise an error describing what went wrong.
  virtual llvm::Expected<lldb::StackFrameListSP>
  GetStackFrames(lldb::StackFrameListSP real_frames) = 0;

protected:
  SyntheticFrameProvider(lldb::ThreadSP thread_sp);

  lldb::ThreadSP m_thread_sp;
};

} // namespace lldb_private

#endif // LLDB_TARGET_SYNTHETICFRAMEPROVIDER_H
