//===-- Editline.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_EDITLINE_H
#define LLDB_HOST_EDITLINE_H

#include "lldb/Host/Config.h"

#include <atomic>
#include <csignal>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "lldb/Host/StreamFile.h"
#include "lldb/Utility/CompletionRequest.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/StringList.h"
#include "lldb/lldb-private.h"

#include "llvm/ADT/FunctionExtras.h"

#if LLDB_ENABLE_REPLXX
#include <replxx.h>
#endif

#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace lldb_private {
namespace line_editor {

using IsInputCompleteCallbackType =
    llvm::unique_function<bool(Editline *, StringList &)>;

using FixIndentationCallbackType =
    llvm::unique_function<int(Editline *, StringList &, int)>;

using SuggestionCallbackType =
    llvm::unique_function<std::optional<std::string>(llvm::StringRef)>;

using CompleteCallbackType = llvm::unique_function<void(CompletionRequest &)>;

using RedrawCallbackType = llvm::unique_function<void()>;

enum class EditorStatus {
  Editing,
  Complete,
  EndOfInput,
  Interrupted
};

enum class CursorLocation {
  BlockStart,
  EditingPrompt,
  EditingCursor,
  BlockEnd
};

enum class HistoryOperation {
  Oldest,
  Older,
  Current,
  Newer,
  Newest
};

} // namespace line_editor

using namespace line_editor;

/// Instances of Editline provide line editing using replxx.
/// Both single- and multi-line editing are supported.
class Editline {
public:
  Editline(const char *editor_name, FILE *input_file,
           lldb::LockableStreamFileSP output_stream_sp,
           lldb::LockableStreamFileSP error_stream_sp, bool color);

  ~Editline();

  static void
  DisplayCompletions(Editline &editline,
                     llvm::ArrayRef<CompletionResult::Completion> results);

  void UseColor(bool use_color);
  void SetPrompt(const char *prompt);
  void SetContinuationPrompt(const char *continuation_prompt);
  void TerminalSizeChanged();
  const char *GetPrompt();
  uint32_t GetCurrentLine();
  bool Interrupt();
  bool Cancel();

  void SetSuggestionCallback(SuggestionCallbackType callback) {
    m_suggestion_callback = std::move(callback);
  }

  void SetRedrawCallback(RedrawCallbackType callback) {
    m_redraw_callback = std::move(callback);
  }

  void SetAutoCompleteCallback(CompleteCallbackType callback) {
    m_completion_callback = std::move(callback);
  }

  void SetIsInputCompleteCallback(IsInputCompleteCallbackType callback) {
    m_is_input_complete_callback = std::move(callback);
  }

  void SetFixIndentationCallback(FixIndentationCallbackType callback,
                                 const char *indent_chars) {
    m_fix_indentation_callback = std::move(callback);
    m_fix_indentation_callback_chars = indent_chars;
  }

  void SetPromptAnsiPrefix(std::string prefix) {
    if (m_color)
      m_prompt_ansi_prefix = std::move(prefix);
    else
      m_prompt_ansi_prefix.clear();
  }

  void SetPromptAnsiSuffix(std::string suffix) {
    if (m_color)
      m_prompt_ansi_suffix = std::move(suffix);
    else
      m_prompt_ansi_suffix.clear();
  }

  void SetSuggestionAnsiPrefix(std::string prefix) {
    if (m_color)
      m_suggestion_ansi_prefix = std::move(prefix);
    else
      m_suggestion_ansi_prefix.clear();
  }

  void SetSuggestionAnsiSuffix(std::string suffix) {
    if (m_color)
      m_suggestion_ansi_suffix = std::move(suffix);
    else
      m_suggestion_ansi_suffix.clear();
  }

  bool GetLine(std::string &line, bool &interrupted);
  bool GetLines(int first_line_number, StringList &lines, bool &interrupted);

  void PrintAsync(lldb::LockableStreamFileSP stream_sp, const char *s,
                  size_t len);

  StringList GetInputAsStringList(int line_count = UINT32_MAX);

  size_t GetTerminalWidth() { return m_terminal_width; }
  size_t GetTerminalHeight() { return m_terminal_height; }

  void Refresh();

private:
  void InitializeReplxx();
  void UpdateTerminalSize();
  std::string PromptForIndex(int line_index);
  void SetCurrentLine(int line_index);
  size_t GetPromptWidth();
  const char *GetHistoryFilePath();

  // replxx callback helpers (static so they match C function pointer type)
  static void CompletionCallback(const char *input,
                                 replxx_completions *completions,
                                 int *context_len, void *user_data);
  static void HintCallback(const char *input, replxx_hints *hints,
                           int *context_len, ReplxxColor *color,
                           void *user_data);
  static ReplxxActionResult ApplySuggestionHandler(int code, void *user_data);

#if LLDB_ENABLE_REPLXX
  ::Replxx *m_replxx = nullptr;
#endif

  std::string m_editor_name;
  FILE *m_input_file;
  lldb::LockableStreamFileSP m_output_stream_sp;
  lldb::LockableStreamFileSP m_error_stream_sp;

  std::vector<std::string> m_input_lines;
  EditorStatus m_editor_status = EditorStatus::Complete;
  int m_terminal_width = 0;
  int m_terminal_height = 0;
  int m_base_line_number = 0;
  unsigned m_current_line_index = 0;
  int m_line_number_digits = 3;

  std::string m_set_prompt;
  std::string m_set_continuation_prompt;
  std::string m_current_prompt;
  std::string m_history_path;

  bool m_color;
  std::string m_prompt_ansi_prefix;
  std::string m_prompt_ansi_suffix;
  std::string m_suggestion_ansi_prefix;
  std::string m_suggestion_ansi_suffix;

  volatile std::sig_atomic_t m_terminal_size_has_changed = 0;

  std::atomic<bool> m_reading{false};
#if !defined(_WIN32)
  pthread_t m_reading_thread = {};
#endif

  IsInputCompleteCallbackType m_is_input_complete_callback;
  FixIndentationCallbackType m_fix_indentation_callback;
  const char *m_fix_indentation_callback_chars = nullptr;
  CompleteCallbackType m_completion_callback;
  SuggestionCallbackType m_suggestion_callback;
  RedrawCallbackType m_redraw_callback;
};

} // namespace lldb_private

#endif // LLDB_HOST_EDITLINE_H
