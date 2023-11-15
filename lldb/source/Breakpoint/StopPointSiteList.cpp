//===-- StopPointSiteList.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/StopPointSiteList.h"
#include "lldb/Breakpoint/BreakpointSite.h"
#include "lldb/Breakpoint/WatchpointResource.h"

#include "lldb/Utility/Stream.h"
#include <algorithm>

using namespace lldb;
using namespace lldb_private;

// Add site to the list.  However, if the element already exists in
// the list, then we don't add it, and return InvalidSiteID.

template <typename StopPointSite>
typename StopPointSite::SiteID
StopPointSiteList<StopPointSite>::Add(const StopPointSiteSP &site) {
  lldb::addr_t site_load_addr = site->GetLoadAddress();
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  typename collection::iterator iter = m_site_list.find(site_load_addr);

  if (iter == m_site_list.end()) {
    m_site_list[site_load_addr] = site;
    return site->GetID();
  } else {
    return UINT32_MAX;
  }
}

template <typename StopPointSite>
bool StopPointSiteList<StopPointSite>::ShouldStop(
    StoppointCallbackContext *context, typename StopPointSite::SiteID site_id) {
  if (StopPointSiteSP site_sp = FindByID(site_id)) {
    // Let the site decide if it should stop here (could not have
    // reached it's target hit count yet, or it could have a callback that
    // decided it shouldn't stop (shared library loads/unloads).
    return site_sp->ShouldStop(context);
  }
  // We should stop here since this site isn't valid anymore or it
  // doesn't exist.
  return true;
}

template <typename StopPointSite>
typename StopPointSite::SiteID
StopPointSiteList<StopPointSite>::FindIDByAddress(lldb::addr_t addr) {
  if (StopPointSiteSP site = FindByAddress(addr))
    return site->GetID();
  return UINT32_MAX;
}

template <typename StopPointSite>
bool StopPointSiteList<StopPointSite>::Remove(
    typename StopPointSite::SiteID site_id) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  typename collection::iterator pos = GetIDIterator(site_id); // Predicate
  if (pos != m_site_list.end()) {
    m_site_list.erase(pos);
    return true;
  }
  return false;
}

template <typename StopPointSite>
bool StopPointSiteList<StopPointSite>::RemoveByAddress(lldb::addr_t address) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  typename collection::iterator pos = m_site_list.find(address);
  if (pos != m_site_list.end()) {
    m_site_list.erase(pos);
    return true;
  }
  return false;
}

template <typename StopPointSite>
typename StopPointSiteList<StopPointSite>::collection::iterator
StopPointSiteList<StopPointSite>::GetIDIterator(
    typename StopPointSite::SiteID site_id) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  auto id_matches = [site_id](const std::pair<addr_t, StopPointSiteSP> s) {
    return site_id == s.second->GetID();
  };
  return std::find_if(m_site_list.begin(),
                      m_site_list.end(), // Search full range
                      id_matches);
}

template <typename StopPointSite>
typename StopPointSiteList<StopPointSite>::collection::const_iterator
StopPointSiteList<StopPointSite>::GetIDConstIterator(
    typename StopPointSite::SiteID site_id) const {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  auto id_matches = [site_id](const std::pair<addr_t, StopPointSiteSP> s) {
    return site_id == s.second->GetID();
  };
  return std::find_if(m_site_list.begin(),
                      m_site_list.end(), // Search full range
                      id_matches);
}

template <typename StopPointSite>
typename StopPointSiteList<StopPointSite>::StopPointSiteSP
StopPointSiteList<StopPointSite>::FindByID(
    typename StopPointSite::SiteID site_id) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  StopPointSiteSP stop_sp;
  typename collection::iterator pos = GetIDIterator(site_id);
  if (pos != m_site_list.end())
    stop_sp = pos->second;

  return stop_sp;
}

template <typename StopPointSite>
const typename StopPointSiteList<StopPointSite>::StopPointSiteSP
StopPointSiteList<StopPointSite>::FindByID(
    typename StopPointSite::SiteID site_id) const {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  StopPointSiteSP stop_sp;
  typename collection::const_iterator pos = GetIDConstIterator(site_id);
  if (pos != m_site_list.end())
    stop_sp = pos->second;

  return stop_sp;
}

template <typename StopPointSite>
typename StopPointSiteList<StopPointSite>::StopPointSiteSP
StopPointSiteList<StopPointSite>::FindByAddress(lldb::addr_t addr) {
  StopPointSiteSP found_sp;
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  typename collection::iterator iter = m_site_list.find(addr);
  if (iter != m_site_list.end())
    found_sp = iter->second;
  return found_sp;
}

// This method is only defined when we're specializing for
// BreakpointSite / BreakpointLocation / Breakpoint.
// Watchpoints don't have a similar structure, they are
// WatchpointResource / Watchpoint
template <>
bool StopPointSiteList<BreakpointSite>::StopPointSiteContainsBreakpoint(
    typename BreakpointSite::SiteID site_id, break_id_t bp_id) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  typename collection::const_iterator pos = GetIDConstIterator(site_id);
  if (pos != m_site_list.end())
    return pos->second->IsBreakpointAtThisSite(bp_id);

  return false;
}

template <typename StopPointSite>
void StopPointSiteList<StopPointSite>::Dump(Stream *s) const {
  s->Printf("%p: ", static_cast<const void *>(this));
  s->Printf("StopPointSiteList with %u ConstituentSites:\n",
            (uint32_t)m_site_list.size());
  s->IndentMore();
  typename collection::const_iterator pos;
  typename collection::const_iterator end = m_site_list.end();
  for (pos = m_site_list.begin(); pos != end; ++pos)
    pos->second->Dump(s);
  s->IndentLess();
}

template <typename StopPointSite>
void StopPointSiteList<StopPointSite>::ForEach(
    std::function<void(StopPointSite *)> const &callback) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  for (auto pair : m_site_list)
    callback(pair.second.get());
}

template <typename StopPointSite>
bool StopPointSiteList<StopPointSite>::FindInRange(
    lldb::addr_t lower_bound, lldb::addr_t upper_bound,
    StopPointSiteList &site_list) const {
  if (lower_bound > upper_bound)
    return false;

  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  typename collection::const_iterator lower, upper, pos;
  lower = m_site_list.lower_bound(lower_bound);
  if (lower == m_site_list.end() || (*lower).first >= upper_bound)
    return false;

  // This is one tricky bit.  The site might overlap the bottom end of
  // the range.  So we grab the site prior to the lower bound, and check
  // that that + its byte size isn't in our range.
  if (lower != m_site_list.begin()) {
    typename collection::const_iterator prev_pos = lower;
    prev_pos--;
    const StopPointSiteSP &prev_site = (*prev_pos).second;
    if (prev_site->GetLoadAddress() + prev_site->GetByteSize() > lower_bound)
      site_list.Add(prev_site);
  }

  upper = m_site_list.upper_bound(upper_bound);

  for (pos = lower; pos != upper; pos++)
    site_list.Add((*pos).second);
  return true;
}

template <typename StopPointSite>
std::vector<typename StopPointSiteList<StopPointSite>::StopPointSiteSP>
StopPointSiteList<StopPointSite>::Sites() {
  std::vector<StopPointSiteSP> sites;
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  typename collection::iterator iter = m_site_list.begin();
  while (iter != m_site_list.end()) {
    sites.push_back(iter->second);
    ++iter;
  }

  return sites;
}

namespace lldb_private {
template class StopPointSiteList<BreakpointSite>;
template class StopPointSiteList<WatchpointResource>;
} // namespace lldb_private
