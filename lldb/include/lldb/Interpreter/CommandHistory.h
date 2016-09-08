//===-- CommandHistory.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandHistory_h_
#define liblldb_CommandHistory_h_

// C Includes
// C++ Includes
#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"
#include "lldb/Host/Mutex.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class CommandHistory {
public:
  CommandHistory();

  ~CommandHistory();

  size_t GetSize() const;

  bool IsEmpty() const;

  const char *FindString(const char *input_str) const;

  const char *GetStringAtIndex(size_t idx) const;

  const char *operator[](size_t idx) const;

  const char *GetRecentmostString() const;

  void AppendString(const std::string &str, bool reject_if_dupe = true);

  void Clear();

  void Dump(Stream &stream, size_t start_idx = 0,
            size_t stop_idx = SIZE_MAX) const;

  static const char g_repeat_char = '!';

private:
  DISALLOW_COPY_AND_ASSIGN(CommandHistory);

  typedef std::vector<std::string> History;
  mutable Mutex m_mutex;
  History m_history;
};

} // namespace lldb_private

#endif // liblldb_CommandHistory_h_
