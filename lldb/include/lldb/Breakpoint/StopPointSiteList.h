//===-- StopPointSiteList.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_BREAKPOINT_STOPPOINTSITELIST_H
#define LLDB_BREAKPOINT_STOPPOINTSITELIST_H

#include <functional>
#include <map>
#include <mutex>

#include <lldb/Utility/Iterable.h>
#include <lldb/Utility/Stream.h>

namespace lldb_private {

template <typename StopPointSite> class StopPointSiteList {
  // At present Process directly accesses the map of StopPointSites so it can
  // do quick lookups into the map (using GetMap).
  // FIXME: Find a better interface for this.
  friend class Process;

public:
  using StopPointSiteSP = std::shared_ptr<StopPointSite>;

  /// Add a site to the list.
  ///
  /// \param[in] site_sp
  ///    A shared pointer to a site being added to the list.
  ///
  /// \return
  ///    The ID of the site in the list.
  typename StopPointSite::SiteID Add(const StopPointSiteSP &site_sp);

  /// Standard Dump routine, doesn't do anything at present.
  /// \param[in] s
  ///     Stream into which to dump the description.
  void Dump(Stream *s) const;

  /// Returns a shared pointer to the site at address \a addr.
  ///
  /// \param[in] addr
  ///     The address to look for.
  ///
  /// \result
  ///     A shared pointer to the site. Nullptr if no site contains
  ///     the address.
  StopPointSiteSP FindByAddress(lldb::addr_t addr);

  /// Returns a shared pointer to the site with id \a site_id.
  ///
  /// \param[in] site_id
  ///   The site ID to seek for.
  ///
  /// \result
  ///   A shared pointer to the site. Nullptr if no matching site.
  StopPointSiteSP FindByID(typename StopPointSite::SiteID site_id);

  /// Returns a shared pointer to the site with id \a site_id -
  /// const version.
  ///
  /// \param[in] site_id
  ///   The site ID to seek for.
  ///
  /// \result
  ///   A shared pointer to the site. Nullptr if no matching site.
  const StopPointSiteSP FindByID(typename StopPointSite::SiteID site_id) const;

  /// Returns the site id to the site at address \a addr.
  ///
  /// \param[in] addr
  ///   The address to match.
  ///
  /// \result
  ///   The ID of the site, or LLDB_INVALID_SITE_ID.
  typename StopPointSite::SiteID FindIDByAddress(lldb::addr_t addr);

  /// Returns whether the BreakpointSite \a site_id has a BreakpointLocation
  /// that is part of Breakpoint \a bp_id.
  ///
  /// NB this is only defined when StopPointSiteList is specialized for
  /// BreakpointSite's.
  ///
  /// \param[in] site_id
  ///   The site id to query.
  ///
  /// \param[in] bp_id
  ///   The breakpoint id to look for in \a site_id's BreakpointLocations.
  ///
  /// \result
  ///   True if \a site_id exists in the site list AND \a bp_id
  ///   is the breakpoint for one of the BreakpointLocations.
  bool StopPointSiteContainsBreakpoint(typename StopPointSite::SiteID,
                                       lldb::break_id_t bp_id);

  void ForEach(std::function<void(StopPointSite *)> const &callback);

  /// Removes the site given by \a site_id from this list.
  ///
  /// \param[in] site_id
  ///   The site ID to remove.
  ///
  /// \result
  ///   \b true if the site \a site_id was in the list.
  bool Remove(typename StopPointSite::SiteID site_id);

  /// Removes the site at address \a addr from this list.
  ///
  /// \param[in] addr
  ///   The address from which to remove a site.
  ///
  /// \result
  ///   \b true if \a addr had a site to remove from the list.
  bool RemoveByAddress(lldb::addr_t addr);

  bool FindInRange(lldb::addr_t lower_bound, lldb::addr_t upper_bound,
                   StopPointSiteList &bp_site_list) const;

  typedef void (*StopPointSiteSPMapFunc)(StopPointSite &site, void *baton);

  /// Enquires of the site on in this list with ID \a site_id
  /// whether we should stop for the constituent or not.
  ///
  /// \param[in] context
  ///    This contains the information about this stop.
  ///
  /// \param[in] site_id
  ///    This site ID that we hit.
  ///
  /// \return
  ///    \b true if we should stop, \b false otherwise.
  bool ShouldStop(StoppointCallbackContext *context,
                  typename StopPointSite::SiteID site_id);

  /// Returns the number of elements in the list.
  ///
  /// \result
  ///   The number of elements.
  size_t GetSize() const {
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    return m_site_list.size();
  }

  bool IsEmpty() const {
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    return m_site_list.empty();
  }

  std::vector<StopPointSiteSP> Sites();

  void Clear() {
    std::lock_guard<std::recursive_mutex> guard(m_mutex);
    m_site_list.clear();
  }

protected:
  typedef std::map<lldb::addr_t, StopPointSiteSP> collection;

  typename collection::iterator
  GetIDIterator(typename StopPointSite::SiteID site_id);

  typename collection::const_iterator
  GetIDConstIterator(typename StopPointSite::SiteID site_id) const;

  mutable std::recursive_mutex m_mutex;
  collection m_site_list; // The site list.
};

} // namespace lldb_private

#endif // LLDB_BREAKPOINT_STOPPOINTSITELIST_H
