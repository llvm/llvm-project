//===-- CommandCompletions.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <sys/stat.h>
#if defined(__APPLE__) || defined(__linux__)
#include <pwd.h>
#endif

// C++ Includes
// Other libraries and framework includes
#include "llvm/ADT/SmallString.h"

// Project includes
#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/CleanUp.h"

#include "llvm/ADT/SmallString.h"

using namespace lldb_private;

CommandCompletions::CommonCompletionElement
    CommandCompletions::g_common_completions[] = {
        {eCustomCompletion, nullptr},
        {eSourceFileCompletion, CommandCompletions::SourceFiles},
        {eDiskFileCompletion, CommandCompletions::DiskFiles},
        {eDiskDirectoryCompletion, CommandCompletions::DiskDirectories},
        {eSymbolCompletion, CommandCompletions::Symbols},
        {eModuleCompletion, CommandCompletions::Modules},
        {eSettingsNameCompletion, CommandCompletions::SettingsNames},
        {ePlatformPluginCompletion, CommandCompletions::PlatformPluginNames},
        {eArchitectureCompletion, CommandCompletions::ArchitectureNames},
        {eVariablePathCompletion, CommandCompletions::VariablePath},
        {eNoCompletion, nullptr} // This one has to be last in the list.
};

bool CommandCompletions::InvokeCommonCompletionCallbacks(
    CommandInterpreter &interpreter, uint32_t completion_mask,
    const char *completion_str, int match_start_point, int max_return_elements,
    SearchFilter *searcher, bool &word_complete, StringList &matches) {
  bool handled = false;

  if (completion_mask & eCustomCompletion)
    return false;

  for (int i = 0;; i++) {
    if (g_common_completions[i].type == eNoCompletion)
      break;
    else if ((g_common_completions[i].type & completion_mask) ==
                 g_common_completions[i].type &&
             g_common_completions[i].callback != nullptr) {
      handled = true;
      g_common_completions[i].callback(interpreter, completion_str,
                                       match_start_point, max_return_elements,
                                       searcher, word_complete, matches);
    }
  }
  return handled;
}

int CommandCompletions::SourceFiles(CommandInterpreter &interpreter,
                                    const char *partial_file_name,
                                    int match_start_point,
                                    int max_return_elements,
                                    SearchFilter *searcher, bool &word_complete,
                                    StringList &matches) {
  word_complete = true;
  // Find some way to switch "include support files..."
  SourceFileCompleter completer(interpreter, false, partial_file_name,
                                match_start_point, max_return_elements,
                                matches);

  if (searcher == nullptr) {
    lldb::TargetSP target_sp = interpreter.GetDebugger().GetSelectedTarget();
    SearchFilterForUnconstrainedSearches null_searcher(target_sp);
    completer.DoCompletion(&null_searcher);
  } else {
    completer.DoCompletion(searcher);
  }
  return matches.GetSize();
}

typedef struct DiskFilesOrDirectoriesBaton {
  const char *remainder;
  char *partial_name_copy;
  bool only_directories;
  bool *saw_directory;
  StringList *matches;
  char *end_ptr;
  size_t baselen;
} DiskFilesOrDirectoriesBaton;

FileSpec::EnumerateDirectoryResult
DiskFilesOrDirectoriesCallback(void *baton, FileSpec::FileType file_type,
                               const FileSpec &spec) {
  const char *name = spec.GetFilename().AsCString();

  const DiskFilesOrDirectoriesBaton *parameters =
      (DiskFilesOrDirectoriesBaton *)baton;
  char *end_ptr = parameters->end_ptr;
  char *partial_name_copy = parameters->partial_name_copy;
  const char *remainder = parameters->remainder;

  // Omit ".", ".." and any . files if the match string doesn't start with .
  if (name[0] == '.') {
    if (name[1] == '\0')
      return FileSpec::eEnumerateDirectoryResultNext;
    else if (name[1] == '.' && name[2] == '\0')
      return FileSpec::eEnumerateDirectoryResultNext;
    else if (remainder[0] != '.')
      return FileSpec::eEnumerateDirectoryResultNext;
  }

  // If we found a directory, we put a "/" at the end of the name.

  if (remainder[0] == '\0' || strstr(name, remainder) == name) {
    if (strlen(name) + parameters->baselen >= PATH_MAX)
      return FileSpec::eEnumerateDirectoryResultNext;

    strcpy(end_ptr, name);

    bool isa_directory = false;
    if (file_type == FileSpec::eFileTypeDirectory)
      isa_directory = true;
    else if (file_type == FileSpec::eFileTypeSymbolicLink) {
      if (FileSpec(partial_name_copy, false).IsDirectory())
        isa_directory = true;
    }

    if (isa_directory) {
      *parameters->saw_directory = true;
      size_t len = strlen(parameters->partial_name_copy);
      partial_name_copy[len] = '/';
      partial_name_copy[len + 1] = '\0';
    }
    if (parameters->only_directories && !isa_directory)
      return FileSpec::eEnumerateDirectoryResultNext;
    parameters->matches->AppendString(partial_name_copy);
  }

  return FileSpec::eEnumerateDirectoryResultNext;
}

static int DiskFilesOrDirectories(const char *partial_file_name,
                                  bool only_directories, bool &saw_directory,
                                  StringList &matches) {
  // I'm going to  use the "glob" function with GLOB_TILDE for user directory
  // expansion.
  // If it is not defined on your host system, you'll need to implement it
  // yourself...

  size_t partial_name_len = strlen(partial_file_name);

  if (partial_name_len >= PATH_MAX)
    return matches.GetSize();

  // This copy of the string will be cut up into the directory part, and the
  // remainder.  end_ptr
  // below will point to the place of the remainder in this string.  Then when
  // we've resolved the
  // containing directory, and opened it, we'll read the directory contents and
  // overwrite the
  // partial_name_copy starting from end_ptr with each of the matches.  Thus we
  // will preserve
  // the form the user originally typed.

  char partial_name_copy[PATH_MAX];
  memcpy(partial_name_copy, partial_file_name, partial_name_len);
  partial_name_copy[partial_name_len] = '\0';

  // We'll need to save a copy of the remainder for comparison, which we do
  // here.
  char remainder[PATH_MAX];

  // end_ptr will point past the last / in partial_name_copy, or if there is no
  // slash to the beginning of the string.
  char *end_ptr;

  end_ptr = strrchr(partial_name_copy, '/');

  // This will store the resolved form of the containing directory
  llvm::SmallString<64> containing_part;

  if (end_ptr == nullptr) {
    // There's no directory.  If the thing begins with a "~" then this is a bare
    // user name.
    if (*partial_name_copy == '~') {
      // Nothing here but the user name.  We could just put a slash on the end,
      // but for completeness sake we'll resolve the user name and only put a
      // slash
      // on the end if it exists.
      llvm::SmallString<64> resolved_username(partial_name_copy);
      FileSpec::ResolveUsername(resolved_username);

      // Not sure how this would happen, a username longer than PATH_MAX?
      // Still...
      if (resolved_username.size() == 0) {
        // The user name didn't resolve, let's look in the password database for
        // matches.
        // The user name database contains duplicates, and is not in
        // alphabetical order, so
        // we'll use a set to manage that for us.
        FileSpec::ResolvePartialUsername(partial_name_copy, matches);
        if (matches.GetSize() > 0)
          saw_directory = true;
        return matches.GetSize();
      } else {
        // The thing exists, put a '/' on the end, and return it...
        // FIXME: complete user names here:
        partial_name_copy[partial_name_len] = '/';
        partial_name_copy[partial_name_len + 1] = '\0';
        matches.AppendString(partial_name_copy);
        saw_directory = true;
        return matches.GetSize();
      }
    } else {
      // The containing part is the CWD, and the whole string is the remainder.
      containing_part = ".";
      strcpy(remainder, partial_name_copy);
      end_ptr = partial_name_copy;
    }
  } else {
    if (end_ptr == partial_name_copy) {
      // We're completing a file or directory in the root volume.
      containing_part = "/";
    } else {
      containing_part.append(partial_name_copy, end_ptr);
    }
    // Push end_ptr past the final "/" and set remainder.
    end_ptr++;
    strcpy(remainder, end_ptr);
  }

  // Look for a user name in the containing part, and if it's there, resolve it
  // and stick the
  // result back into the containing_part:

  if (*partial_name_copy == '~') {
    FileSpec::ResolveUsername(containing_part);
    // User name doesn't exist, we're not getting any further...
    if (containing_part.empty())
      return matches.GetSize();
  }

  // Okay, containing_part is now the directory we want to open and look for
  // files:

  size_t baselen = end_ptr - partial_name_copy;

  DiskFilesOrDirectoriesBaton parameters;
  parameters.remainder = remainder;
  parameters.partial_name_copy = partial_name_copy;
  parameters.only_directories = only_directories;
  parameters.saw_directory = &saw_directory;
  parameters.matches = &matches;
  parameters.end_ptr = end_ptr;
  parameters.baselen = baselen;

  FileSpec::EnumerateDirectory(containing_part.c_str(), true, true, true,
                               DiskFilesOrDirectoriesCallback, &parameters);

  return matches.GetSize();
}

int CommandCompletions::DiskFiles(CommandInterpreter &interpreter,
                                  const char *partial_file_name,
                                  int match_start_point,
                                  int max_return_elements,
                                  SearchFilter *searcher, bool &word_complete,
                                  StringList &matches) {
  int ret_val =
      DiskFilesOrDirectories(partial_file_name, false, word_complete, matches);
  word_complete = !word_complete;
  return ret_val;
}

int CommandCompletions::DiskDirectories(
    CommandInterpreter &interpreter, const char *partial_file_name,
    int match_start_point, int max_return_elements, SearchFilter *searcher,
    bool &word_complete, StringList &matches) {
  int ret_val =
      DiskFilesOrDirectories(partial_file_name, true, word_complete, matches);
  word_complete = false;
  return ret_val;
}

int CommandCompletions::Modules(CommandInterpreter &interpreter,
                                const char *partial_file_name,
                                int match_start_point, int max_return_elements,
                                SearchFilter *searcher, bool &word_complete,
                                StringList &matches) {
  word_complete = true;
  ModuleCompleter completer(interpreter, partial_file_name, match_start_point,
                            max_return_elements, matches);

  if (searcher == nullptr) {
    lldb::TargetSP target_sp = interpreter.GetDebugger().GetSelectedTarget();
    SearchFilterForUnconstrainedSearches null_searcher(target_sp);
    completer.DoCompletion(&null_searcher);
  } else {
    completer.DoCompletion(searcher);
  }
  return matches.GetSize();
}

int CommandCompletions::Symbols(CommandInterpreter &interpreter,
                                const char *partial_file_name,
                                int match_start_point, int max_return_elements,
                                SearchFilter *searcher, bool &word_complete,
                                StringList &matches) {
  word_complete = true;
  SymbolCompleter completer(interpreter, partial_file_name, match_start_point,
                            max_return_elements, matches);

  if (searcher == nullptr) {
    lldb::TargetSP target_sp = interpreter.GetDebugger().GetSelectedTarget();
    SearchFilterForUnconstrainedSearches null_searcher(target_sp);
    completer.DoCompletion(&null_searcher);
  } else {
    completer.DoCompletion(searcher);
  }
  return matches.GetSize();
}

int CommandCompletions::SettingsNames(
    CommandInterpreter &interpreter, const char *partial_setting_name,
    int match_start_point, int max_return_elements, SearchFilter *searcher,
    bool &word_complete, StringList &matches) {
  // Cache the full setting name list
  static StringList g_property_names;
  if (g_property_names.GetSize() == 0) {
    // Generate the full setting name list on demand
    lldb::OptionValuePropertiesSP properties_sp(
        interpreter.GetDebugger().GetValueProperties());
    if (properties_sp) {
      StreamString strm;
      properties_sp->DumpValue(nullptr, strm, OptionValue::eDumpOptionName);
      const std::string &str = strm.GetString();
      g_property_names.SplitIntoLines(str.c_str(), str.size());
    }
  }

  size_t exact_matches_idx = SIZE_MAX;
  const size_t num_matches = g_property_names.AutoComplete(
      partial_setting_name, matches, exact_matches_idx);
  word_complete = exact_matches_idx != SIZE_MAX;
  return num_matches;
}

int CommandCompletions::PlatformPluginNames(
    CommandInterpreter &interpreter, const char *partial_name,
    int match_start_point, int max_return_elements, SearchFilter *searcher,
    bool &word_complete, lldb_private::StringList &matches) {
  const uint32_t num_matches =
      PluginManager::AutoCompletePlatformName(partial_name, matches);
  word_complete = num_matches == 1;
  return num_matches;
}

int CommandCompletions::ArchitectureNames(
    CommandInterpreter &interpreter, const char *partial_name,
    int match_start_point, int max_return_elements, SearchFilter *searcher,
    bool &word_complete, lldb_private::StringList &matches) {
  const uint32_t num_matches = ArchSpec::AutoComplete(partial_name, matches);
  word_complete = num_matches == 1;
  return num_matches;
}

int CommandCompletions::VariablePath(
    CommandInterpreter &interpreter, const char *partial_name,
    int match_start_point, int max_return_elements, SearchFilter *searcher,
    bool &word_complete, lldb_private::StringList &matches) {
  return Variable::AutoComplete(interpreter.GetExecutionContext(), partial_name,
                                matches, word_complete);
}

CommandCompletions::Completer::Completer(CommandInterpreter &interpreter,
                                         const char *completion_str,
                                         int match_start_point,
                                         int max_return_elements,
                                         StringList &matches)
    : m_interpreter(interpreter), m_completion_str(completion_str),
      m_match_start_point(match_start_point),
      m_max_return_elements(max_return_elements), m_matches(matches) {}

CommandCompletions::Completer::~Completer() = default;

//----------------------------------------------------------------------
// SourceFileCompleter
//----------------------------------------------------------------------

CommandCompletions::SourceFileCompleter::SourceFileCompleter(
    CommandInterpreter &interpreter, bool include_support_files,
    const char *completion_str, int match_start_point, int max_return_elements,
    StringList &matches)
    : CommandCompletions::Completer(interpreter, completion_str,
                                    match_start_point, max_return_elements,
                                    matches),
      m_include_support_files(include_support_files), m_matching_files() {
  FileSpec partial_spec(m_completion_str.c_str(), false);
  m_file_name = partial_spec.GetFilename().GetCString();
  m_dir_name = partial_spec.GetDirectory().GetCString();
}

Searcher::Depth CommandCompletions::SourceFileCompleter::GetDepth() {
  return eDepthCompUnit;
}

Searcher::CallbackReturn
CommandCompletions::SourceFileCompleter::SearchCallback(SearchFilter &filter,
                                                        SymbolContext &context,
                                                        Address *addr,
                                                        bool complete) {
  if (context.comp_unit != nullptr) {
    if (m_include_support_files) {
      FileSpecList supporting_files = context.comp_unit->GetSupportFiles();
      for (size_t sfiles = 0; sfiles < supporting_files.GetSize(); sfiles++) {
        const FileSpec &sfile_spec =
            supporting_files.GetFileSpecAtIndex(sfiles);
        const char *sfile_file_name = sfile_spec.GetFilename().GetCString();
        const char *sfile_dir_name = sfile_spec.GetFilename().GetCString();
        bool match = false;
        if (m_file_name && sfile_file_name &&
            strstr(sfile_file_name, m_file_name) == sfile_file_name)
          match = true;
        if (match && m_dir_name && sfile_dir_name &&
            strstr(sfile_dir_name, m_dir_name) != sfile_dir_name)
          match = false;

        if (match) {
          m_matching_files.AppendIfUnique(sfile_spec);
        }
      }
    } else {
      const char *cur_file_name = context.comp_unit->GetFilename().GetCString();
      const char *cur_dir_name = context.comp_unit->GetDirectory().GetCString();

      bool match = false;
      if (m_file_name && cur_file_name &&
          strstr(cur_file_name, m_file_name) == cur_file_name)
        match = true;

      if (match && m_dir_name && cur_dir_name &&
          strstr(cur_dir_name, m_dir_name) != cur_dir_name)
        match = false;

      if (match) {
        m_matching_files.AppendIfUnique(context.comp_unit);
      }
    }
  }
  return Searcher::eCallbackReturnContinue;
}

size_t
CommandCompletions::SourceFileCompleter::DoCompletion(SearchFilter *filter) {
  filter->Search(*this);
  // Now convert the filelist to completions:
  for (size_t i = 0; i < m_matching_files.GetSize(); i++) {
    m_matches.AppendString(
        m_matching_files.GetFileSpecAtIndex(i).GetFilename().GetCString());
  }
  return m_matches.GetSize();
}

//----------------------------------------------------------------------
// SymbolCompleter
//----------------------------------------------------------------------

static bool regex_chars(const char comp) {
  return (comp == '[' || comp == ']' || comp == '(' || comp == ')' ||
          comp == '{' || comp == '}' || comp == '+' || comp == '.' ||
          comp == '*' || comp == '|' || comp == '^' || comp == '$' ||
          comp == '\\' || comp == '?');
}

CommandCompletions::SymbolCompleter::SymbolCompleter(
    CommandInterpreter &interpreter, const char *completion_str,
    int match_start_point, int max_return_elements, StringList &matches)
    : CommandCompletions::Completer(interpreter, completion_str,
                                    match_start_point, max_return_elements,
                                    matches) {
  std::string regex_str;
  if (completion_str && completion_str[0]) {
    regex_str.append("^");
    regex_str.append(completion_str);
  } else {
    // Match anything since the completion string is empty
    regex_str.append(".");
  }
  std::string::iterator pos =
      find_if(regex_str.begin() + 1, regex_str.end(), regex_chars);
  while (pos < regex_str.end()) {
    pos = regex_str.insert(pos, '\\');
    pos = find_if(pos + 2, regex_str.end(), regex_chars);
  }
  m_regex.Compile(regex_str);
}

Searcher::Depth CommandCompletions::SymbolCompleter::GetDepth() {
  return eDepthModule;
}

Searcher::CallbackReturn CommandCompletions::SymbolCompleter::SearchCallback(
    SearchFilter &filter, SymbolContext &context, Address *addr,
    bool complete) {
  if (context.module_sp) {
    SymbolContextList sc_list;
    const bool include_symbols = true;
    const bool include_inlines = true;
    const bool append = true;
    context.module_sp->FindFunctions(m_regex, include_symbols, include_inlines,
                                     append, sc_list);

    SymbolContext sc;
    // Now add the functions & symbols to the list - only add if unique:
    for (uint32_t i = 0; i < sc_list.GetSize(); i++) {
      if (sc_list.GetContextAtIndex(i, sc)) {
        ConstString func_name = sc.GetFunctionName(Mangled::ePreferDemangled);
        if (!func_name.IsEmpty())
          m_match_set.insert(func_name);
      }
    }
  }
  return Searcher::eCallbackReturnContinue;
}

size_t CommandCompletions::SymbolCompleter::DoCompletion(SearchFilter *filter) {
  filter->Search(*this);
  collection::iterator pos = m_match_set.begin(), end = m_match_set.end();
  for (pos = m_match_set.begin(); pos != end; pos++)
    m_matches.AppendString((*pos).GetCString());

  return m_matches.GetSize();
}

//----------------------------------------------------------------------
// ModuleCompleter
//----------------------------------------------------------------------
CommandCompletions::ModuleCompleter::ModuleCompleter(
    CommandInterpreter &interpreter, const char *completion_str,
    int match_start_point, int max_return_elements, StringList &matches)
    : CommandCompletions::Completer(interpreter, completion_str,
                                    match_start_point, max_return_elements,
                                    matches) {
  FileSpec partial_spec(m_completion_str.c_str(), false);
  m_file_name = partial_spec.GetFilename().GetCString();
  m_dir_name = partial_spec.GetDirectory().GetCString();
}

Searcher::Depth CommandCompletions::ModuleCompleter::GetDepth() {
  return eDepthModule;
}

Searcher::CallbackReturn CommandCompletions::ModuleCompleter::SearchCallback(
    SearchFilter &filter, SymbolContext &context, Address *addr,
    bool complete) {
  if (context.module_sp) {
    const char *cur_file_name =
        context.module_sp->GetFileSpec().GetFilename().GetCString();
    const char *cur_dir_name =
        context.module_sp->GetFileSpec().GetDirectory().GetCString();

    bool match = false;
    if (m_file_name && cur_file_name &&
        strstr(cur_file_name, m_file_name) == cur_file_name)
      match = true;

    if (match && m_dir_name && cur_dir_name &&
        strstr(cur_dir_name, m_dir_name) != cur_dir_name)
      match = false;

    if (match) {
      m_matches.AppendString(cur_file_name);
    }
  }
  return Searcher::eCallbackReturnContinue;
}

size_t CommandCompletions::ModuleCompleter::DoCompletion(SearchFilter *filter) {
  filter->Search(*this);
  return m_matches.GetSize();
}
