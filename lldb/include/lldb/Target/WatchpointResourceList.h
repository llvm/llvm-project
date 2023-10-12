//===-- WatchpointResourceList.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_WATCHPOINTRESOURCELIST_H
#define LLDB_TARGET_WATCHPOINTRESOURCELIST_H

#include "lldb/Utility/Iterable.h"
#include "lldb/lldb-private.h"
#include "lldb/lldb-public.h"

#include <mutex>
#include <vector>

namespace lldb_private {

class WatchpointResourceList {

public:
  // Constructors and Destructors
  WatchpointResourceList();

  ~WatchpointResourceList();

  /// Get the number of WatchpointResources that are available.
  ///
  /// \return
  ///     The number of WatchpointResources that are stored in the
  ///     WatchpointResourceList.
  uint32_t GetSize();

  /// Get the WatchpointResource at a given index.
  ///
  /// \param [in] idx
  ///     The index of the resource.
  /// \return
  ///     The WatchpointResource at that index number.
  lldb::WatchpointResourceSP GetResourceAtIndex(uint32_t idx);

  /// Remove a WatchpointResource from the list.
  ///
  /// The WatchpointResource must have already been disabled in the
  /// Process; this method only removes it from the list.
  ///
  /// \param [in] wp_resource_sp
  ///     The WatchpointResource to remove.
  void RemoveWatchpointResource(lldb::WatchpointResourceSP wp_resource_sp);

  typedef std::vector<lldb::WatchpointResourceSP> collection;
  typedef LockingAdaptedIterable<collection, lldb::WatchpointResourceSP,
                                 vector_adapter, std::mutex>
      WatchpointResourceIterable;

  /// Iterate over the list of WatchpointResources.
  ///
  /// \return
  ///     An Iterable object which can be used to loop over the resources
  ///     that exist.
  WatchpointResourceIterable Resources() {
    return WatchpointResourceIterable(m_resources, m_mutex);
  }

  /// Clear out the list of resources from the WatchpointResourceList
  void Clear();

  /// Add a WatchpointResource to the WatchpointResourceList.
  ///
  /// \param [in] resource
  ///     A WatchpointResource to be added.
  void AddResource(lldb::WatchpointResourceSP resource_sp);

  std::mutex &GetMutex();

private:
  collection m_resources;
  std::mutex m_mutex;
};

} // namespace lldb_private

#endif // LLDB_TARGET_WATCHPOINTRESOURCELIST_H
