//===-- WatchpointResource.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_BREAKPOINT_WATCHPOINTRESOURCE_H
#define LLDB_BREAKPOINT_WATCHPOINTRESOURCE_H

#include "lldb/Utility/Iterable.h"
#include "lldb/lldb-public.h"

#include <mutex>
#include <set>

namespace lldb_private {

class WatchpointResource
    : public std::enable_shared_from_this<WatchpointResource> {

public:
  // Constructors and Destructors
  WatchpointResource(lldb::addr_t addr, size_t size, bool read, bool write);

  ~WatchpointResource();

  lldb::addr_t GetAddress() const;

  size_t GetByteSize() const;

  bool WatchpointResourceRead() const;

  bool WatchpointResourceWrite() const;

  void SetType(bool read, bool write);

  typedef std::vector<lldb::WatchpointSP> WatchpointCollection;
  typedef LockingAdaptedIterable<WatchpointCollection, lldb::WatchpointSP,
                                 vector_adapter, std::recursive_mutex>
      WatchpointIterable;

  /// Iterate over the watchpoint owners for this resource
  ///
  /// \return
  ///     An Iterable object which can be used to loop over the watchpoints
  ///     that are owners of this resource.
  WatchpointIterable Owners() {
    return WatchpointIterable(m_owners, m_owners_mutex);
  }

  /// The "Owners" are the watchpoints that share this resource.
  /// The method adds the \a owner to this resource's owner list.
  ///
  /// \param[in] owner
  ///    \a owner is the Wachpoint to add.
  void AddOwner(const lldb::WatchpointSP &owner);

  /// The method removes the owner at \a owner from this watchpoint
  /// resource.
  void RemoveOwner(lldb::WatchpointSP &owner);

  /// This method returns the number of Watchpoints currently using
  /// watchpoint resource.
  ///
  /// \return
  ///    The number of owners.
  size_t GetNumberOfOwners();

  /// This method returns the Watchpoint at index \a index using this
  /// Resource.  The owners are listed ordinally from 0 to
  /// GetNumberOfOwners() - 1 so you can use this method to iterate over the
  /// owners.
  ///
  /// \param[in] idx
  ///     The index in the list of owners for which you wish the owner location.
  ///
  /// \return
  ///    The Watchpoint at that index.
  lldb::WatchpointSP GetOwnerAtIndex(size_t idx);

  /// Check if the owners includes a watchpoint.
  ///
  /// \param[in] wp_sp
  ///     The WatchpointSP to search for.
  ///
  /// \result
  ///     true if this resource's owners includes the watchpoint.
  bool OwnersContains(lldb::WatchpointSP &wp_sp);

  /// Check if the owners includes a watchpoint.
  ///
  /// \param[in] wp
  ///     The Watchpoint to search for.
  ///
  /// \result
  ///     true if this resource's owners includes the watchpoint.
  bool OwnersContains(const lldb_private::Watchpoint *wp);

  /// This method copies the watchpoint resource's owners into a new collection.
  /// It does this while the owners mutex is locked.
  ///
  /// \param[out] out_collection
  ///    The BreakpointLocationCollection into which to put the owners
  ///    of this breakpoint site.
  ///
  /// \return
  ///    The number of elements copied into out_collection.
  size_t CopyOwnersList(WatchpointCollection &out_collection);

  // The ID of the WatchpointResource is set by the WatchpointResourceList
  // when the Resource has been set in the inferior and is being added
  // to the List, in an attempt to match the hardware watchpoint register
  // ordering.  If a Process can correctly identify the hardware watchpoint
  // register index when it has created the Resource, it may initialize it
  // before it is inserted in the WatchpointResourceList.
  void SetID(lldb::wp_resource_id_t);

  lldb::wp_resource_id_t GetID() const;

  bool Contains(lldb::addr_t addr);

protected:
  // The StopInfoWatchpoint knows when it is processing a hit for a thread for
  // a site, so let it be the one to manage setting the location hit count once
  // and only once.
  friend class StopInfoWatchpoint;

  void BumpHitCounts();

private:
  lldb::wp_resource_id_t m_id;

  // start address & size aligned & expanded to be a valid watchpoint
  // memory granule on this target.
  lldb::addr_t m_addr;
  size_t m_size;

  bool m_watch_read;  // true if we stop when the watched data is read from
  bool m_watch_write; // true if we stop when the watched data is written to

  // The watchpoints using this WatchpointResource.
  WatchpointCollection m_owners;

  std::recursive_mutex
      m_owners_mutex; ///< This mutex protects the owners collection.

  WatchpointResource(const WatchpointResource &) = delete;
  const WatchpointResource &operator=(const WatchpointResource &) = delete;
};

} // namespace lldb_private

#endif // LLDB_BREAKPOINT_WATCHPOINTRESOURCE_H
