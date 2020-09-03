//===-- FileSpecList.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_FILESPECLIST_H
#define LLDB_CORE_FILESPECLIST_H
#if defined(__cplusplus)

#include "lldb/Utility/FileSpec.h"

#include <vector>

#include <stddef.h>

namespace lldb_private {
class Stream;

/// \class FileSpecList FileSpecList.h "lldb/Core/FileSpecList.h"
/// A file collection class.
///
/// A class that contains a mutable list of FileSpec objects.
class FileSpecList {
public:
  typedef std::vector<FileSpec> collection;
  typedef collection::const_iterator const_iterator;

  /// Default constructor.
  ///
  /// Initialize this object with an empty file list.
  FileSpecList();

  /// Copy constructor.
  FileSpecList(const FileSpecList &rhs) = default;

  /// Move constructor
  FileSpecList(FileSpecList &&rhs) = default;

  /// Initialize this object from a vector of FileSpecs
  FileSpecList(std::vector<FileSpec> &&rhs) : m_files(std::move(rhs)) {}

  /// Destructor.
  ~FileSpecList();

  /// Assignment operator.
  ///
  /// Replace the file list in this object with the file list from \a rhs.
  ///
  /// \param[in] rhs
  ///     A file list object to copy.
  ///
  /// \return
  ///     A const reference to this object.
  FileSpecList &operator=(const FileSpecList &rhs) = default;

  /// Move-assignment operator.
  FileSpecList &operator=(FileSpecList &&rhs) = default;

  /// Append a FileSpec object to the list.
  ///
  /// Appends \a file to the end of the file list.
  ///
  /// \param[in] file
  ///     A new file to append to this file list.
  void Append(const FileSpec &file);

  /// Append a FileSpec object if unique.
  ///
  /// Appends \a file to the end of the file list if it doesn't already exist
  /// in the file list.
  ///
  /// \param[in] file
  ///     A new file to append to this file list.
  ///
  /// \return
  ///     \b true if the file was appended, \b false otherwise.
  bool AppendIfUnique(const FileSpec &file);

  /// Inserts a new FileSpec into the FileSpecList constructed in-place with
  /// the given arguments.
  ///
  /// \param[in] args
  ///     Arguments to create the FileSpec
  template <class... Args> void EmplaceBack(Args &&... args) {
    m_files.emplace_back(std::forward<Args>(args)...);
  }

  /// Clears the file list.
  void Clear();

  /// Dumps the file list to the supplied stream pointer "s".
  ///
  /// \param[in] s
  ///     The stream that will be used to dump the object description.
  void Dump(Stream *s, const char *separator_cstr = "\n") const;

  /// Find a file index.
  ///
  /// Find the index of the file in the file spec list that matches \a file
  /// starting \a idx entries into the file spec list.
  ///
  /// \param[in] idx
  ///     An index into the file list.
  ///
  /// \param[in] file
  ///     The file specification to search for.
  ///
  /// \param[in] full
  ///     Should FileSpec::Equal be called with "full" true or false.
  ///
  /// \return
  ///     The index of the file that matches \a file if it is found,
  ///     else UINT32_MAX is returned.
  size_t FindFileIndex(size_t idx, const FileSpec &file, bool full) const;

  /// Get file at index.
  ///
  /// Gets a file from the file list. If \a idx is not a valid index, an empty
  /// FileSpec object will be returned. The file objects that are returned can
  /// be tested using FileSpec::operator void*().
  ///
  /// \param[in] idx
  ///     An index into the file list.
  ///
  /// \return
  ///     A copy of the FileSpec object at index \a idx. If \a idx
  ///     is out of range, then an empty FileSpec object will be
  ///     returned.
  const FileSpec &GetFileSpecAtIndex(size_t idx) const;

  /// Get file specification pointer at index.
  ///
  /// Gets a file from the file list. The file objects that are returned can
  /// be tested using FileSpec::operator void*().
  ///
  /// \param[in] idx
  ///     An index into the file list.
  ///
  /// \return
  ///     A pointer to a contained FileSpec object at index \a idx.
  ///     If \a idx is out of range, then an NULL is returned.
  const FileSpec *GetFileSpecPointerAtIndex(size_t idx) const;

  /// Get the memory cost of this object.
  ///
  /// Return the size in bytes that this object takes in memory. This returns
  /// the size in bytes of this object, not any shared string values it may
  /// refer to.
  ///
  /// \return
  ///     The number of bytes that this object occupies in memory.
  ///
  /// \see ConstString::StaticMemorySize ()
  size_t MemorySize() const;

  bool IsEmpty() const { return m_files.empty(); }

  /// Get the number of files in the file list.
  ///
  /// \return
  ///     The number of files in the file spec list.
  size_t GetSize() const;

  bool Insert(size_t idx, const FileSpec &file) {
    if (idx < m_files.size()) {
      m_files.insert(m_files.begin() + idx, file);
      return true;
    } else if (idx == m_files.size()) {
      m_files.push_back(file);
      return true;
    }
    return false;
  }

  bool Replace(size_t idx, const FileSpec &file) {
    if (idx < m_files.size()) {
      m_files[idx] = file;
      return true;
    }
    return false;
  }

  bool Remove(size_t idx) {
    if (idx < m_files.size()) {
      m_files.erase(m_files.begin() + idx);
      return true;
    }
    return false;
  }

  static size_t GetFilesMatchingPartialPath(const char *path, bool dir_okay,
                                            FileSpecList &matches);

  const_iterator begin() const { return m_files.begin(); }
  const_iterator end() const { return m_files.end(); }

protected:
  collection m_files; ///< A collection of FileSpec objects.
};

} // namespace lldb_private

#endif // #if defined(__cplusplus)
#endif // LLDB_CORE_FILESPECLIST_H
