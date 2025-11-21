//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_SYNTHETICFRAMEPROVIDER_H
#define LLDB_TARGET_SYNTHETICFRAMEPROVIDER_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/Target/StackFrameList.h"
#include "lldb/Target/ThreadSpec.h"
#include "lldb/Utility/ScriptedMetadata.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <vector>

namespace lldb_private {

/// This struct contains the metadata needed to instantiate a frame provider
/// and optional filters to control which threads it applies to.
struct SyntheticFrameProviderDescriptor {
  /// Metadata for instantiating the provider (e.g. script class name and args).
  lldb::ScriptedMetadataSP scripted_metadata_sp;

  /// Optional list of thread specifications to which this provider applies.
  /// If empty, the provider applies to all threads. A thread matches if it
  /// satisfies ANY of the specs in this vector (OR logic).
  std::vector<ThreadSpec> thread_specs;

  SyntheticFrameProviderDescriptor() = default;

  SyntheticFrameProviderDescriptor(lldb::ScriptedMetadataSP metadata_sp)
      : scripted_metadata_sp(metadata_sp) {}

  SyntheticFrameProviderDescriptor(lldb::ScriptedMetadataSP metadata_sp,
                                   const std::vector<ThreadSpec> &specs)
      : scripted_metadata_sp(metadata_sp), thread_specs(specs) {}

  /// Get the name of this descriptor (the scripted class name).
  llvm::StringRef GetName() const {
    return scripted_metadata_sp ? scripted_metadata_sp->GetClassName() : "";
  }

  /// Check if this descriptor applies to the given thread.
  bool AppliesToThread(Thread &thread) const {
    // If no thread specs specified, applies to all threads.
    if (thread_specs.empty())
      return true;

    // Check if the thread matches any of the specs (OR logic).
    for (const auto &spec : thread_specs) {
      if (spec.ThreadPassesBasicTests(thread))
        return true;
    }
    return false;
  }

  /// Check if this descriptor has valid metadata for script-based providers.
  bool IsValid() const { return scripted_metadata_sp != nullptr; }

  void Dump(Stream *s) const;
};

/// Base class for all synthetic frame providers.
///
/// Synthetic frame providers allow modifying or replacing the stack frames
/// shown for a thread. This is useful for:
/// - Providing frames for custom calling conventions or languages.
/// - Reconstructing missing frames from crash dumps or core files.
/// - Adding diagnostic or synthetic frames for debugging.
/// - Visualizing state machines or async execution contexts.
class SyntheticFrameProvider : public PluginInterface {
public:
  /// Try to create a SyntheticFrameProvider instance for the given input
  /// frames and descriptor.
  ///
  /// This method iterates through all registered SyntheticFrameProvider
  /// plugins and returns the first one that can handle the given descriptor.
  ///
  /// \param[in] input_frames
  ///     The input stack frame list that this provider will transform.
  ///     This could be real unwound frames or output from another provider.
  ///
  /// \param[in] descriptor
  ///     The descriptor containing metadata for the provider.
  ///
  /// \return
  ///     A shared pointer to a SyntheticFrameProvider if one could be created,
  ///     otherwise an \a llvm::Error.
  static llvm::Expected<lldb::SyntheticFrameProviderSP>
  CreateInstance(lldb::StackFrameListSP input_frames,
                 const SyntheticFrameProviderDescriptor &descriptor);

  /// Try to create a SyntheticFrameProvider instance for the given input
  /// frames using a specific C++ plugin.
  ///
  /// This method directly invokes a specific SyntheticFrameProvider plugin
  /// by name, bypassing the descriptor-based plugin iteration. This is useful
  /// for C++ plugins that don't require scripted metadata.
  ///
  /// \param[in] input_frames
  ///     The input stack frame list that this provider will transform.
  ///     This could be real unwound frames or output from another provider.
  ///
  /// \param[in] plugin_name
  ///     The name of the plugin to use for creating the provider.
  ///
  /// \param[in] thread_specs
  ///     Optional list of thread specifications to which this provider applies.
  ///     If empty, the provider applies to all threads.
  ///
  /// \return
  ///     A shared pointer to a SyntheticFrameProvider if one could be created,
  ///     otherwise an \a llvm::Error.
  static llvm::Expected<lldb::SyntheticFrameProviderSP>
  CreateInstance(lldb::StackFrameListSP input_frames,
                 llvm::StringRef plugin_name,
                 const std::vector<ThreadSpec> &thread_specs = {});

  ~SyntheticFrameProvider() override;

  /// Get a single stack frame at the specified index.
  ///
  /// This method is called lazily - frames are only created when requested.
  /// The provider can access its input frames via GetInputFrames() if needed.
  ///
  /// \param[in] idx
  ///     The index of the frame to create.
  ///
  /// \return
  ///     An Expected containing the StackFrameSP if successful. Returns an
  ///     error when the index is beyond the last frame to signal the end of
  ///     the frame list.
  virtual llvm::Expected<lldb::StackFrameSP> GetFrameAtIndex(uint32_t idx) = 0;

  /// Get the thread associated with this provider.
  Thread &GetThread() { return m_input_frames->GetThread(); }

  /// Get the input frames that this provider transforms.
  lldb::StackFrameListSP GetInputFrames() const { return m_input_frames; }

protected:
  SyntheticFrameProvider(lldb::StackFrameListSP input_frames);

  lldb::StackFrameListSP m_input_frames;
};

} // namespace lldb_private

#endif // LLDB_TARGET_SYNTHETICFRAMEPROVIDER_H
