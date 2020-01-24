//===-- CommandCompletions.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"

#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/TildeExpressionResolver.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

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
    CompletionRequest &request, SearchFilter *searcher) {
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
      g_common_completions[i].callback(interpreter, request, searcher);
    }
  }
  return handled;
}

void CommandCompletions::SourceFiles(CommandInterpreter &interpreter,
                                     CompletionRequest &request,
                                     SearchFilter *searcher) {
  // Find some way to switch "include support files..."
  SourceFileCompleter completer(interpreter, false, request);

  if (searcher == nullptr) {
    lldb::TargetSP target_sp = interpreter.GetDebugger().GetSelectedTarget();
    SearchFilterForUnconstrainedSearches null_searcher(target_sp);
    completer.DoCompletion(&null_searcher);
  } else {
    completer.DoCompletion(searcher);
  }
}

static void DiskFilesOrDirectories(const llvm::Twine &partial_name,
                                   bool only_directories,
                                   CompletionRequest &request,
                                   TildeExpressionResolver &Resolver) {
  llvm::SmallString<256> CompletionBuffer;
  llvm::SmallString<256> Storage;
  partial_name.toVector(CompletionBuffer);

  if (CompletionBuffer.size() >= PATH_MAX)
    return;

  namespace path = llvm::sys::path;

  llvm::StringRef SearchDir;
  llvm::StringRef PartialItem;

  if (CompletionBuffer.startswith("~")) {
    llvm::StringRef Buffer(CompletionBuffer);
    size_t FirstSep =
        Buffer.find_if([](char c) { return path::is_separator(c); });

    llvm::StringRef Username = Buffer.take_front(FirstSep);
    llvm::StringRef Remainder;
    if (FirstSep != llvm::StringRef::npos)
      Remainder = Buffer.drop_front(FirstSep + 1);

    llvm::SmallString<256> Resolved;
    if (!Resolver.ResolveExact(Username, Resolved)) {
      // We couldn't resolve it as a full username.  If there were no slashes
      // then this might be a partial username.   We try to resolve it as such
      // but after that, we're done regardless of any matches.
      if (FirstSep == llvm::StringRef::npos) {
        llvm::StringSet<> MatchSet;
        Resolver.ResolvePartial(Username, MatchSet);
        for (const auto &S : MatchSet) {
          Resolved = S.getKey();
          path::append(Resolved, path::get_separator());
          request.AddCompletion(Resolved, "", CompletionMode::Partial);
        }
      }
      return;
    }

    // If there was no trailing slash, then we're done as soon as we resolve
    // the expression to the correct directory.  Otherwise we need to continue
    // looking for matches within that directory.
    if (FirstSep == llvm::StringRef::npos) {
      // Make sure it ends with a separator.
      path::append(CompletionBuffer, path::get_separator());
      request.AddCompletion(CompletionBuffer, "", CompletionMode::Partial);
      return;
    }

    // We want to keep the form the user typed, so we special case this to
    // search in the fully resolved directory, but CompletionBuffer keeps the
    // unmodified form that the user typed.
    Storage = Resolved;
    llvm::StringRef RemainderDir = path::parent_path(Remainder);
    if (!RemainderDir.empty()) {
      // Append the remaining path to the resolved directory.
      Storage.append(path::get_separator());
      Storage.append(RemainderDir);
    }
    SearchDir = Storage;
  } else {
    SearchDir = path::parent_path(CompletionBuffer);
  }

  size_t FullPrefixLen = CompletionBuffer.size();

  PartialItem = path::filename(CompletionBuffer);

  // path::filename() will return "." when the passed path ends with a
  // directory separator. We have to filter those out, but only when the
  // "." doesn't come from the completion request itself.
  if (PartialItem == "." && path::is_separator(CompletionBuffer.back()))
    PartialItem = llvm::StringRef();

  if (SearchDir.empty()) {
    llvm::sys::fs::current_path(Storage);
    SearchDir = Storage;
  }
  assert(!PartialItem.contains(path::get_separator()));

  // SearchDir now contains the directory to search in, and Prefix contains the
  // text we want to match against items in that directory.

  FileSystem &fs = FileSystem::Instance();
  std::error_code EC;
  llvm::vfs::directory_iterator Iter = fs.DirBegin(SearchDir, EC);
  llvm::vfs::directory_iterator End;
  for (; Iter != End && !EC; Iter.increment(EC)) {
    auto &Entry = *Iter;
    llvm::ErrorOr<llvm::vfs::Status> Status = fs.GetStatus(Entry.path());

    if (!Status)
      continue;

    auto Name = path::filename(Entry.path());

    // Omit ".", ".."
    if (Name == "." || Name == ".." || !Name.startswith(PartialItem))
      continue;

    bool is_dir = Status->isDirectory();

    // If it's a symlink, then we treat it as a directory as long as the target
    // is a directory.
    if (Status->isSymlink()) {
      FileSpec symlink_filespec(Entry.path());
      FileSpec resolved_filespec;
      auto error = fs.ResolveSymbolicLink(symlink_filespec, resolved_filespec);
      if (error.Success())
        is_dir = fs.IsDirectory(symlink_filespec);
    }

    if (only_directories && !is_dir)
      continue;

    // Shrink it back down so that it just has the original prefix the user
    // typed and remove the part of the name which is common to the located
    // item and what the user typed.
    CompletionBuffer.resize(FullPrefixLen);
    Name = Name.drop_front(PartialItem.size());
    CompletionBuffer.append(Name);

    if (is_dir) {
      path::append(CompletionBuffer, path::get_separator());
    }

    CompletionMode mode =
        is_dir ? CompletionMode::Partial : CompletionMode::Normal;
    request.AddCompletion(CompletionBuffer, "", mode);
  }
}

static void DiskFilesOrDirectories(const llvm::Twine &partial_name,
                                   bool only_directories, StringList &matches,
                                   TildeExpressionResolver &Resolver) {
  CompletionResult result;
  std::string partial_name_str = partial_name.str();
  CompletionRequest request(partial_name_str, partial_name_str.size(), result);
  DiskFilesOrDirectories(partial_name, only_directories, request, Resolver);
  result.GetMatches(matches);
}

static void DiskFilesOrDirectories(CompletionRequest &request,
                                   bool only_directories) {
  StandardTildeExpressionResolver resolver;
  DiskFilesOrDirectories(request.GetCursorArgumentPrefix(), only_directories,
                         request, resolver);
}

void CommandCompletions::DiskFiles(CommandInterpreter &interpreter,
                                   CompletionRequest &request,
                                   SearchFilter *searcher) {
  DiskFilesOrDirectories(request, /*only_dirs*/ false);
}

void CommandCompletions::DiskFiles(const llvm::Twine &partial_file_name,
                                   StringList &matches,
                                   TildeExpressionResolver &Resolver) {
  DiskFilesOrDirectories(partial_file_name, false, matches, Resolver);
}

void CommandCompletions::DiskDirectories(CommandInterpreter &interpreter,
                                         CompletionRequest &request,
                                         SearchFilter *searcher) {
  DiskFilesOrDirectories(request, /*only_dirs*/ true);
}

void CommandCompletions::DiskDirectories(const llvm::Twine &partial_file_name,
                                         StringList &matches,
                                         TildeExpressionResolver &Resolver) {
  DiskFilesOrDirectories(partial_file_name, true, matches, Resolver);
}

void CommandCompletions::Modules(CommandInterpreter &interpreter,
                                 CompletionRequest &request,
                                 SearchFilter *searcher) {
  ModuleCompleter completer(interpreter, request);

  if (searcher == nullptr) {
    lldb::TargetSP target_sp = interpreter.GetDebugger().GetSelectedTarget();
    SearchFilterForUnconstrainedSearches null_searcher(target_sp);
    completer.DoCompletion(&null_searcher);
  } else {
    completer.DoCompletion(searcher);
  }
}

void CommandCompletions::Symbols(CommandInterpreter &interpreter,
                                 CompletionRequest &request,
                                 SearchFilter *searcher) {
  SymbolCompleter completer(interpreter, request);

  if (searcher == nullptr) {
    lldb::TargetSP target_sp = interpreter.GetDebugger().GetSelectedTarget();
    SearchFilterForUnconstrainedSearches null_searcher(target_sp);
    completer.DoCompletion(&null_searcher);
  } else {
    completer.DoCompletion(searcher);
  }
}

void CommandCompletions::SettingsNames(CommandInterpreter &interpreter,
                                       CompletionRequest &request,
                                       SearchFilter *searcher) {
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

  for (const std::string &s : g_property_names)
    request.TryCompleteCurrentArg(s);
}

void CommandCompletions::PlatformPluginNames(CommandInterpreter &interpreter,
                                             CompletionRequest &request,
                                             SearchFilter *searcher) {
  PluginManager::AutoCompletePlatformName(request.GetCursorArgumentPrefix(),
                                          request);
}

void CommandCompletions::ArchitectureNames(CommandInterpreter &interpreter,
                                           CompletionRequest &request,
                                           SearchFilter *searcher) {
  ArchSpec::AutoComplete(request);
}

void CommandCompletions::VariablePath(CommandInterpreter &interpreter,
                                      CompletionRequest &request,
                                      SearchFilter *searcher) {
  Variable::AutoComplete(interpreter.GetExecutionContext(), request);
}

CommandCompletions::Completer::Completer(CommandInterpreter &interpreter,
                                         CompletionRequest &request)
    : m_interpreter(interpreter), m_request(request) {}

CommandCompletions::Completer::~Completer() = default;

// SourceFileCompleter

CommandCompletions::SourceFileCompleter::SourceFileCompleter(
    CommandInterpreter &interpreter, bool include_support_files,
    CompletionRequest &request)
    : CommandCompletions::Completer(interpreter, request),
      m_include_support_files(include_support_files), m_matching_files() {
  FileSpec partial_spec(m_request.GetCursorArgumentPrefix());
  m_file_name = partial_spec.GetFilename().GetCString();
  m_dir_name = partial_spec.GetDirectory().GetCString();
}

lldb::SearchDepth CommandCompletions::SourceFileCompleter::GetDepth() {
  return lldb::eSearchDepthCompUnit;
}

Searcher::CallbackReturn
CommandCompletions::SourceFileCompleter::SearchCallback(SearchFilter &filter,
                                                        SymbolContext &context,
                                                        Address *addr) {
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
      const char *cur_file_name =
          context.comp_unit->GetPrimaryFile().GetFilename().GetCString();
      const char *cur_dir_name =
          context.comp_unit->GetPrimaryFile().GetDirectory().GetCString();

      bool match = false;
      if (m_file_name && cur_file_name &&
          strstr(cur_file_name, m_file_name) == cur_file_name)
        match = true;

      if (match && m_dir_name && cur_dir_name &&
          strstr(cur_dir_name, m_dir_name) != cur_dir_name)
        match = false;

      if (match) {
        m_matching_files.AppendIfUnique(context.comp_unit->GetPrimaryFile());
      }
    }
  }
  return Searcher::eCallbackReturnContinue;
}

void CommandCompletions::SourceFileCompleter::DoCompletion(
    SearchFilter *filter) {
  filter->Search(*this);
  // Now convert the filelist to completions:
  for (size_t i = 0; i < m_matching_files.GetSize(); i++) {
    m_request.AddCompletion(
        m_matching_files.GetFileSpecAtIndex(i).GetFilename().GetCString());
  }
}

// SymbolCompleter

static bool regex_chars(const char comp) {
  return llvm::StringRef("[](){}+.*|^$\\?").contains(comp);
}

CommandCompletions::SymbolCompleter::SymbolCompleter(
    CommandInterpreter &interpreter, CompletionRequest &request)
    : CommandCompletions::Completer(interpreter, request) {
  std::string regex_str;
  if (!m_request.GetCursorArgumentPrefix().empty()) {
    regex_str.append("^");
    regex_str.append(m_request.GetCursorArgumentPrefix());
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
  m_regex = RegularExpression(regex_str);
}

lldb::SearchDepth CommandCompletions::SymbolCompleter::GetDepth() {
  return lldb::eSearchDepthModule;
}

Searcher::CallbackReturn CommandCompletions::SymbolCompleter::SearchCallback(
    SearchFilter &filter, SymbolContext &context, Address *addr) {
  if (context.module_sp) {
    SymbolContextList sc_list;
    const bool include_symbols = true;
    const bool include_inlines = true;
    context.module_sp->FindFunctions(m_regex, include_symbols, include_inlines,
                                     sc_list);

    SymbolContext sc;
    // Now add the functions & symbols to the list - only add if unique:
    for (uint32_t i = 0; i < sc_list.GetSize(); i++) {
      if (sc_list.GetContextAtIndex(i, sc)) {
        ConstString func_name = sc.GetFunctionName(Mangled::ePreferDemangled);
        // Ensure that the function name matches the regex. This is more than a
        // sanity check. It is possible that the demangled function name does
        // not start with the prefix, for example when it's in an anonymous
        // namespace.
        if (!func_name.IsEmpty() && m_regex.Execute(func_name.GetStringRef()))
          m_match_set.insert(func_name);
      }
    }
  }
  return Searcher::eCallbackReturnContinue;
}

void CommandCompletions::SymbolCompleter::DoCompletion(SearchFilter *filter) {
  filter->Search(*this);
  collection::iterator pos = m_match_set.begin(), end = m_match_set.end();
  for (pos = m_match_set.begin(); pos != end; pos++)
    m_request.AddCompletion((*pos).GetCString());
}

// ModuleCompleter
CommandCompletions::ModuleCompleter::ModuleCompleter(
    CommandInterpreter &interpreter, CompletionRequest &request)
    : CommandCompletions::Completer(interpreter, request) {
  FileSpec partial_spec(m_request.GetCursorArgumentPrefix());
  m_file_name = partial_spec.GetFilename().GetCString();
  m_dir_name = partial_spec.GetDirectory().GetCString();
}

lldb::SearchDepth CommandCompletions::ModuleCompleter::GetDepth() {
  return lldb::eSearchDepthModule;
}

Searcher::CallbackReturn CommandCompletions::ModuleCompleter::SearchCallback(
    SearchFilter &filter, SymbolContext &context, Address *addr) {
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
      m_request.AddCompletion(cur_file_name);
    }
  }
  return Searcher::eCallbackReturnContinue;
}

void CommandCompletions::ModuleCompleter::DoCompletion(SearchFilter *filter) {
  filter->Search(*this);
}
