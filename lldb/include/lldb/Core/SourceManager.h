//===-- SourceManager.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_SOURCEMANAGER_H
#define LLDB_CORE_SOURCEMANAGER_H

#include "lldb/Utility/Checksum.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-forward.h"

#include "llvm/Support/Chrono.h"
#include "llvm/Support/RWMutex.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lldb_private {
class RegularExpression;
class Stream;
class SymbolContextList;
class Target;

class SourceManager {
public:
  class File {
    friend bool operator==(const SourceManager::File &lhs,
                           const SourceManager::File &rhs);

  public:
    File(lldb::SupportFileSP support_file_sp, lldb::TargetSP target_sp);
    File(lldb::SupportFileSP support_file_sp, lldb::DebuggerSP debugger_sp);

    bool ModificationTimeIsStale() const;
    bool PathRemappingIsStale() const;

    size_t DisplaySourceLines(uint32_t line, std::optional<size_t> column,
                              uint32_t context_before, uint32_t context_after,
                              Stream *s);
    void FindLinesMatchingRegex(RegularExpression &regex, uint32_t start_line,
                                uint32_t end_line,
                                std::vector<uint32_t> &match_lines);

    bool GetLine(uint32_t line_no, std::string &buffer);

    uint32_t GetLineOffset(uint32_t line);

    bool LineIsValid(uint32_t line);

    lldb::SupportFileSP GetSupportFile() const {
      assert(m_support_file_sp && "SupportFileSP must always be valid");
      return m_support_file_sp;
    }

    uint32_t GetSourceMapModificationID() const { return m_source_map_mod_id; }

    const char *PeekLineData(uint32_t line);

    uint32_t GetLineLength(uint32_t line, bool include_newline_chars);

    uint32_t GetNumLines();

    llvm::sys::TimePoint<> GetTimestamp() const { return m_mod_time; }

    const Checksum &GetChecksum() const { return m_checksum; }

    llvm::once_flag &GetChecksumWarningOnceFlag() {
      return m_checksum_warning_once_flag;
    }

  protected:
    /// Set file and update modification time.
    void SetSupportFile(lldb::SupportFileSP support_file_sp);

    bool CalculateLineOffsets(uint32_t line = UINT32_MAX);

    /// The support file. If the target has source mappings, this might be
    /// different from the original support file passed to the constructor.
    lldb::SupportFileSP m_support_file_sp;

    /// Keep track of the on-disk checksum.
    Checksum m_checksum;

    /// Once flag for emitting a checksum mismatch warning.
    llvm::once_flag m_checksum_warning_once_flag;

    // Keep the modification time that this file data is valid for
    llvm::sys::TimePoint<> m_mod_time;

    // If the target uses path remappings, be sure to clear our notion of a
    // source file if the path modification ID changes
    uint32_t m_source_map_mod_id = 0;
    lldb::DataBufferSP m_data_sp;
    typedef std::vector<uint32_t> LineOffsets;
    LineOffsets m_offsets;
    lldb::DebuggerWP m_debugger_wp;
    lldb::TargetWP m_target_wp;

  private:
    void CommonInitializer(lldb::SupportFileSP support_file_sp,
                           lldb::TargetSP target_sp);
  };

  typedef std::shared_ptr<File> FileSP;

  /// The SourceFileCache class separates the source manager from the cache of
  /// source files. There is one source manager per Target but both the Debugger
  /// and the Process have their own source caches.
  ///
  /// The SourceFileCache just handles adding, storing, removing and looking up
  /// source files. The caching policies are implemented in
  /// SourceManager::GetFile.
  class SourceFileCache {
  public:
    SourceFileCache() = default;
    ~SourceFileCache() = default;

    void AddSourceFile(const FileSpec &file_spec, FileSP file_sp);
    void RemoveSourceFile(const FileSP &file_sp);

    FileSP FindSourceFile(const FileSpec &file_spec) const;

    // Removes all elements from the cache.
    void Clear() { m_file_cache.clear(); }

    void Dump(Stream &stream) const;

  private:
    void AddSourceFileImpl(const FileSpec &file_spec, FileSP file_sp);

    typedef std::map<FileSpec, FileSP> FileCache;
    FileCache m_file_cache;

    mutable llvm::sys::RWMutex m_mutex;
  };

  /// A source manager can be made with a valid Target, in which case it can use
  /// the path remappings to find source files that are not in their build
  /// locations.  Without a target it won't be able to do this.
  /// @{
  SourceManager(const lldb::DebuggerSP &debugger_sp);
  SourceManager(const lldb::TargetSP &target_sp);
  /// @}

  ~SourceManager();

  FileSP GetLastFile() { return GetFile(m_last_support_file_sp); }

  size_t DisplaySourceLinesWithLineNumbers(
      lldb::SupportFileSP support_file_sp, uint32_t line, uint32_t column,
      uint32_t context_before, uint32_t context_after,
      const char *current_line_cstr, Stream *s,
      const SymbolContextList *bp_locs = nullptr);

  // This variant uses the last file we visited.
  size_t DisplaySourceLinesWithLineNumbersUsingLastFile(
      uint32_t start_line, uint32_t count, uint32_t curr_line, uint32_t column,
      const char *current_line_cstr, Stream *s,
      const SymbolContextList *bp_locs = nullptr);

  size_t DisplayMoreWithLineNumbers(Stream *s, uint32_t count, bool reverse,
                                    const SymbolContextList *bp_locs = nullptr);

  bool SetDefaultFileAndLine(lldb::SupportFileSP support_file_sp,
                             uint32_t line);

  struct SupportFileAndLine {
    lldb::SupportFileSP support_file_sp;
    uint32_t line;
    SupportFileAndLine(lldb::SupportFileSP support_file_sp, uint32_t line)
        : support_file_sp(support_file_sp), line(line) {}
  };

  std::optional<SupportFileAndLine> GetDefaultFileAndLine();

  bool DefaultFileAndLineSet() {
    return (GetFile(m_last_support_file_sp).get() != nullptr);
  }

  void FindLinesMatchingRegex(lldb::SupportFileSP support_file_sp,
                              RegularExpression &regex, uint32_t start_line,
                              uint32_t end_line,
                              std::vector<uint32_t> &match_lines);

  FileSP GetFile(lldb::SupportFileSP support_file_sp);

protected:
  lldb::SupportFileSP m_last_support_file_sp;
  uint32_t m_last_line;
  uint32_t m_last_count;
  bool m_default_set;
  lldb::TargetWP m_target_wp;
  lldb::DebuggerWP m_debugger_wp;

private:
  SourceManager(const SourceManager &) = delete;
  const SourceManager &operator=(const SourceManager &) = delete;
};

bool operator==(const SourceManager::File &lhs, const SourceManager::File &rhs);

} // namespace lldb_private

#endif // LLDB_CORE_SOURCEMANAGER_H
