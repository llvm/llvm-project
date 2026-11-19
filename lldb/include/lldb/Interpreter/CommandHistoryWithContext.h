//===-- CommandHistoryWithContext.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_COMMANDHISTORYWITHCONTEXT_H
#define LLDB_INTERPRETER_COMMANDHISTORYWITHCONTEXT_H

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>

#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

/// The context in which a command was executed.
///
/// This is recorded alongside every command so that autosuggestions can prefer
/// commands that were previously run in a context similar to the current one.
struct CommandExecutionContext {
  /// Basename of the target's executable (e.g. "a.out"). Empty if there is no
  /// target.
  std::string target_name;
  /// Basename of the source file at the current stop location. Empty if there
  /// is no execution context with source information.
  std::string file;
  /// Line number at the current stop location, or 0 if unknown.
  uint32_t line = 0;
  /// Base name of the function at the current stop location. Empty if there is
  /// no execution context.
  std::string function_name;
  /// Current working directory of LLDB when the command was executed.
  std::string working_directory;
};

/// A command history that records the CommandExecutionContext in which each
/// command was executed and persists to a JSON file.
///
/// Each distinct command is stored once, together with how many times it has
/// been invoked and the context of its most recent invocation. At most
/// GetMaxEntries() commands are retained, both in memory and on disk. When the
/// limit is exceeded, the least frequently invoked command is evicted, breaking
/// ties by evicting the oldest of them. The recorded context is used to rank
/// autosuggestions (see FindSuggestion).
class CommandHistoryWithContext {
public:
  /// \param history_file
  ///     The JSON file used to persist the history. May be an empty FileSpec,
  ///     in which case the history is kept in memory only.
  explicit CommandHistoryWithContext(FileSpec history_file);

  /// The maximum number of commands retained in the history.
  static constexpr size_t GetMaxEntries() { return 200; }

  /// Record that \p command was executed in \p context and persist the updated
  /// history to disk.
  ///
  /// If \p command was seen before, its invocation count is incremented, its
  /// recorded context is refreshed to \p context, and it becomes the most
  /// recently used entry. Otherwise a new entry is added, evicting the least
  /// frequently invoked existing entry (oldest first on ties) if the history is
  /// full.
  void AppendCommand(llvm::StringRef command,
                     const CommandExecutionContext &context);

  /// Find a suggestion that completes \p line, or std::nullopt if none exists.
  ///
  /// Only history entries whose command starts with \p line are considered.
  /// Among those, the following preference order is applied relative to the
  /// \p current context:
  ///   1. the most recent command run at the same source file and line, else
  ///   2. the most recent command run in the same function, else
  ///   3. the most recent command run against the same target.
  /// If none of these context matches apply (e.g. there is no execution
  /// context), the most recent command that starts with \p line is returned.
  ///
  /// The returned string is the remainder of the matching command after
  /// \p line, i.e. the text to append to what the user already typed.
  std::optional<std::string>
  FindSuggestion(llvm::StringRef line,
                 const CommandExecutionContext &current) const;

  /// Load the history from the JSON file, replacing the in-memory history. A
  /// missing or malformed file is treated as an empty history.
  void Load();

  /// Write the history to the JSON file.
  void Save() const;

private:
  struct Entry {
    std::string command;
    CommandExecutionContext context;
    /// The number of times this command has been invoked.
    uint64_t invocation_count = 1;
  };

  /// Implementation of Save() that assumes \c m_mutex is already held.
  void SaveLocked() const;

  mutable std::recursive_mutex m_mutex;
  /// Oldest command at the front, most recent at the back.
  std::deque<Entry> m_entries;
  FileSpec m_history_file;
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_COMMANDHISTORYWITHCONTEXT_H
