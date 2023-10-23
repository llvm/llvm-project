//===-- WatchpointCollection.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_WATCHPOINT_WATCHPOINTCOLLECTION_H
#define LLDB_WATCHPOINT_WATCHPOINTCOLLECTION_H

#include <mutex>
#include <vector>

#include "lldb/Utility/Iterable.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class WatchpointCollection {
public:
  WatchpointCollection();

  ~WatchpointCollection();

  WatchpointCollection &operator=(const WatchpointCollection &rhs);

  /// Add the watchpoint \a wp_sp to the list.
  ///
  /// \param[in] wp_sp
  ///     Shared pointer to the watchpoint that will get added
  ///     to the list.
  void Add(const lldb::WatchpointSP &wp_sp);

  /// Removes the watchpoint given by \a wp_sp from this
  /// list.
  ///
  /// \param[in] wp_sp
  ///     The watchpoint to remove.
  ///
  /// \result
  ///     \b true if the watchpoint was removed.
  bool Remove(lldb::WatchpointSP &wp_sp);

  /// Returns a shared pointer to the watchpoint with index
  /// \a i.
  ///
  /// \param[in] i
  ///     The watchpoint index to seek for.
  ///
  /// \result
  ///     A shared pointer to the watchpoint.  May return
  ///     empty shared pointer if the index is out of bounds.
  lldb::WatchpointSP GetByIndex(size_t i);

  /// Returns a shared pointer to the watchpoint with index
  /// \a i, const version.
  ///
  /// \param[in] i
  ///     The watchpoint index to seek for.
  ///
  /// \result
  ///     A shared pointer to the watchpoint.  May return
  ///     empty shared pointer if the index is out of bounds.
  const lldb::WatchpointSP GetByIndex(size_t i) const;

  /// Returns if the collection includes a WatchpointSP.
  ///
  /// \param[in] wp_sp
  ///     The WatchpointSP to search for.
  ///
  /// \result
  ///     true if this collection includes the WatchpointSP.
  bool Contains(const lldb::WatchpointSP &wp_sp);

  /// Returns if the collection includes a Watchpoint.
  ///
  /// \param[in] wp
  ///     The Watchpoint to search for.
  ///
  /// \result
  ///     true if this collection includes the WatchpointSP.
  bool Contains(const Watchpoint *wp);

  /// Returns the number of elements in this watchpoint list.
  ///
  /// \result
  ///     The number of elements.
  size_t GetSize() const { return m_collection.size(); }

  /// Clear the watchpoint list.
  void Clear();

private:
  // For WatchpointCollection only

  typedef std::vector<lldb::WatchpointSP> collection;
  typedef collection::iterator iterator;
  typedef collection::const_iterator const_iterator;

  collection m_collection;
  mutable std::mutex m_collection_mutex;

public:
  typedef AdaptedIterable<collection, lldb::WatchpointSP, vector_adapter>
      WatchpointCollectionIterable;
  WatchpointCollectionIterable Watchpoints() {
    return WatchpointCollectionIterable(m_collection);
  }
};

} // namespace lldb_private

#endif // LLDB_WATCHPOINT_WATCHPOINTCOLLECTION_H
