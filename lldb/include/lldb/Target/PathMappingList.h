//===-- PathMappingList.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_PATHMAPPINGLIST_H
#define LLDB_TARGET_PATHMAPPINGLIST_H

#include <map>
#include <vector>
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Status.h"

namespace lldb_private {

class PathMappingList {
public:
  typedef void (*ChangedCallback)(const PathMappingList &path_list,
                                  void *baton);

  // Constructors and Destructors
  PathMappingList();

  PathMappingList(ChangedCallback callback, void *callback_baton);

  PathMappingList(const PathMappingList &rhs);

  ~PathMappingList();

  const PathMappingList &operator=(const PathMappingList &rhs);

  void Append(ConstString path, ConstString replacement,
              bool notify);

  void Append(const PathMappingList &rhs, bool notify);

  void Clear(bool notify);

  // By default, dump all pairs.
  void Dump(Stream *s, int pair_index = -1);

  bool IsEmpty() const { return m_pairs.empty(); }

  size_t GetSize() const { return m_pairs.size(); }

  bool GetPathsAtIndex(uint32_t idx, ConstString &path,
                       ConstString &new_path) const;

  void Insert(ConstString path, ConstString replacement,
              uint32_t insert_idx, bool notify);

  bool Remove(size_t index, bool notify);

  bool Remove(ConstString path, bool notify);

  bool Replace(ConstString path, ConstString replacement,
               bool notify);

  bool Replace(ConstString path, ConstString replacement,
               uint32_t index, bool notify);
  bool RemapPath(ConstString path, ConstString &new_path) const;

  /// Remaps a source file given \a path into \a new_path.
  ///
  /// Remaps \a path if any source remappings match. This function
  /// does NOT stat the file system so it can be used in tight loops
  /// where debug info is being parsed.
  ///
  /// \param[in] path
  ///     The original source file path to try and remap.
  ///
  /// \return
  ///     The remapped filespec that may or may not exist on disk.
  llvm::Optional<FileSpec> RemapPath(llvm::StringRef path) const;
  bool RemapPath(const char *, std::string &) const = delete;

  bool ReverseRemapPath(const FileSpec &file, FileSpec &fixed) const;

  /// Finds a source file given a file spec using the path remappings.
  ///
  /// Tries to resolve \a orig_spec by checking the path remappings.
  /// It makes sure the file exists by checking with the file system,
  /// so this call can be expensive if the remappings are on a network
  /// or are even on the local file system, so use this function
  /// sparingly (not in a tight debug info parsing loop).
  ///
  /// \param[in] orig_spec
  ///     The original source file path to try and remap.
  ///
  /// \param[out] new_spec
  ///     The newly remapped filespec that is guaranteed to exist.
  ///
  /// \return
  ///     /b true if \a orig_spec was successfully located and
  ///     \a new_spec is filled in with an existing file spec,
  ///     \b false otherwise.
  bool FindFile(const FileSpec &orig_spec, FileSpec &new_spec) const;

  uint32_t FindIndexForPath(ConstString path) const;

  uint32_t GetModificationID() const { return m_mod_id; }

protected:
  typedef std::pair<ConstString, ConstString> pair;
  typedef std::vector<pair> collection;
  typedef collection::iterator iterator;
  typedef collection::const_iterator const_iterator;

  iterator FindIteratorForPath(ConstString path);

  const_iterator FindIteratorForPath(ConstString path) const;

  collection m_pairs;
  ChangedCallback m_callback = nullptr;
  void *m_callback_baton = nullptr;
  uint32_t m_mod_id = 0; // Incremented anytime anything is added or removed.
};

} // namespace lldb_private

#endif // LLDB_TARGET_PATHMAPPINGLIST_H
