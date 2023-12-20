//===-- WatchpointResourceList.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_BREAKPOINT_WATCHPOINTRESOURCELIST_H
#define LLDB_BREAKPOINT_WATCHPOINTRESOURCELIST_H

#include "lldb/Utility/Iterable.h"
#include "lldb/lldb-private.h"
#include "lldb/lldb-public.h"

#include <mutex>
#include <vector>

namespace lldb_private {

class WatchpointResourceList {

public:
  WatchpointResourceList();

  ~WatchpointResourceList();

  /// Add a WatchpointResource to the list.
  ///
  /// \param[in] wp_res_sp
  ///    A shared pointer to a breakpoint site being added to the list.
  ///
  /// \return
  ///    The ID of the BreakpointSite in the list.
  lldb::wp_resource_id_t Add(const lldb::WatchpointResourceSP &wp_res_sp);

  /// Removes the watchpoint resource given by \a id from this list.
  ///
  /// \param[in] id
  ///   The watchpoint resource to remove.
  ///
  /// \result
  ///   \b true if the watchpoint resource \a id was in the list.
  bool Remove(lldb::wp_resource_id_t id);

  /// Removes the watchpoint resource containing address \a addr from this list.
  ///
  /// \param[in] addr
  ///   The address from which to remove a watchpoint resource.
  ///
  /// \result
  ///   \b true if \a addr had a watchpoint resource to remove from the list.
  bool RemoveByAddress(lldb::addr_t addr);

  /// Returns a shared pointer to the watchpoint resource which contains
  /// \a addr.
  ///
  /// \param[in] addr
  ///     The address to look for.
  ///
  /// \result
  ///     A shared pointer to the watchpoint resource. May contain a nullptr
  ///     pointer if no watchpoint site exists with a matching address.
  lldb::WatchpointResourceSP FindByAddress(lldb::addr_t addr);

  /// Returns a shared pointer to the watchpoint resource which is owned
  /// by \a wp_sp.
  ///
  /// \param[in] wp_sp
  ///     The WatchpointSP to look for.
  ///
  /// \result
  ///     A shared pointer to the watchpoint resource. May contain a nullptr
  ///     pointer if no watchpoint site exists
  lldb::WatchpointResourceSP FindByWatchpointSP(lldb::WatchpointSP &wp_sp);

  /// Returns a shared pointer to the watchpoint resource which is owned
  /// by \a wp.
  ///
  /// \param[in] wp
  ///     The Watchpoint to look for.
  ///
  /// \result
  ///     A shared pointer to the watchpoint resource. May contain a nullptr
  ///     pointer if no watchpoint site exists
  lldb::WatchpointResourceSP
  FindByWatchpoint(const lldb_private::Watchpoint *wp);

  /// Returns a shared pointer to the watchpoint resource which has hardware
  /// index \a id.  Some Process plugins may not have access to the actual
  /// hardware watchpoint register number used for a WatchpointResource, so
  /// the wp_resource_id_t may not be correctly tracking the target's wp
  /// register target.
  ///
  /// \param[in] id
  ///     The hardware resource index to search for.
  ///
  /// \result
  ///     A shared pointer to the watchpoint resource. May contain a nullptr
  ///     pointer if no watchpoint site exists with a matching id.
  lldb::WatchpointResourceSP FindByID(lldb::wp_resource_id_t id);

  ///
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

  std::mutex &GetMutex();

private:
  collection m_resources;
  std::mutex m_mutex;
};

} // namespace lldb_private

#endif // LLDB_BREAKPOINT_WATCHPOINTRESOURCELIST_H
