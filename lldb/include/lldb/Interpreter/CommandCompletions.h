//===-- CommandCompletions.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_CommandCompletions_h_
#define lldb_CommandCompletions_h_

// C Includes
// C++ Includes
#include <set>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/lldb-private.h"

#include "llvm/ADT/Twine.h"

namespace lldb_private {
class TildeExpressionResolver;
class CommandCompletions {
public:
  //----------------------------------------------------------------------
  // This is the command completion callback that is used to complete the
  // argument of the option
  // it is bound to (in the OptionDefinition table below).  Return the total
  // number of matches.
  //----------------------------------------------------------------------
  typedef int (*CompletionCallback)(
      CommandInterpreter &interpreter,
      llvm::StringRef completion_str, // This is the argument we are completing
      int match_start_point,   // This is the point in the list of matches that
                               // you should start returning elements
      int max_return_elements, // This is the number of matches requested.
      lldb_private::SearchFilter
          *searcher, // A search filter to limit the search...
      bool &word_complete,
      lldb_private::StringList &matches); // The array of matches we return.
  typedef enum {
    eNoCompletion = 0u,
    eSourceFileCompletion = (1u << 0),
    eDiskFileCompletion = (1u << 1),
    eDiskDirectoryCompletion = (1u << 2),
    eSymbolCompletion = (1u << 3),
    eModuleCompletion = (1u << 4),
    eSettingsNameCompletion = (1u << 5),
    ePlatformPluginCompletion = (1u << 6),
    eArchitectureCompletion = (1u << 7),
    eVariablePathCompletion = (1u << 8),
    // This item serves two purposes.  It is the last element in the enum,
    // so you can add custom enums starting from here in your Option class.
    // Also if you & in this bit the base code will not process the option.
    eCustomCompletion = (1u << 9)
  } CommonCompletionTypes;

  struct CommonCompletionElement {
    uint32_t type;
    CompletionCallback callback;
  };

  static bool InvokeCommonCompletionCallbacks(
      CommandInterpreter &interpreter, uint32_t completion_mask,
      llvm::StringRef completion_str, int match_start_point,
      int max_return_elements, SearchFilter *searcher, bool &word_complete,
      StringList &matches);

  //----------------------------------------------------------------------
  // These are the generic completer functions:
  //----------------------------------------------------------------------
  static int DiskFiles(CommandInterpreter &interpreter,
                       llvm::StringRef partial_file_name, int match_start_point,
                       int max_return_elements, SearchFilter *searcher,
                       bool &word_complete, StringList &matches);

  static int DiskFiles(const llvm::Twine &partial_file_name,
                       StringList &matches, TildeExpressionResolver &Resolver);

  static int DiskDirectories(CommandInterpreter &interpreter,
                             llvm::StringRef partial_file_name,
                             int match_start_point, int max_return_elements,
                             SearchFilter *searcher, bool &word_complete,
                             StringList &matches);

  static int DiskDirectories(const llvm::Twine &partial_file_name,
                             StringList &matches,
                             TildeExpressionResolver &Resolver);

  static int SourceFiles(CommandInterpreter &interpreter,
                         llvm::StringRef partial_file_name,
                         int match_start_point, int max_return_elements,
                         SearchFilter *searcher, bool &word_complete,
                         StringList &matches);

  static int Modules(CommandInterpreter &interpreter,
                     llvm::StringRef partial_file_name, int match_start_point,
                     int max_return_elements, SearchFilter *searcher,
                     bool &word_complete, lldb_private::StringList &matches);

  static int Symbols(CommandInterpreter &interpreter,
                     llvm::StringRef partial_file_name, int match_start_point,
                     int max_return_elements, SearchFilter *searcher,
                     bool &word_complete, lldb_private::StringList &matches);

  static int SettingsNames(CommandInterpreter &interpreter,
                           llvm::StringRef partial_file_name,
                           int match_start_point, int max_return_elements,
                           SearchFilter *searcher, bool &word_complete,
                           lldb_private::StringList &matches);

  static int PlatformPluginNames(CommandInterpreter &interpreter,
                                 llvm::StringRef partial_file_name,
                                 int match_start_point, int max_return_elements,
                                 SearchFilter *searcher, bool &word_complete,
                                 lldb_private::StringList &matches);

  static int ArchitectureNames(CommandInterpreter &interpreter,
                               llvm::StringRef partial_file_name,
                               int match_start_point, int max_return_elements,
                               SearchFilter *searcher, bool &word_complete,
                               lldb_private::StringList &matches);

  static int VariablePath(CommandInterpreter &interpreter,
                          llvm::StringRef partial_file_name,
                          int match_start_point, int max_return_elements,
                          SearchFilter *searcher, bool &word_complete,
                          lldb_private::StringList &matches);

  //----------------------------------------------------------------------
  // The Completer class is a convenient base class for building searchers
  // that go along with the SearchFilter passed to the standard Completer
  // functions.
  //----------------------------------------------------------------------
  class Completer : public Searcher {
  public:
    Completer(CommandInterpreter &interpreter, llvm::StringRef completion_str,
              int match_start_point, int max_return_elements,
              StringList &matches);

    ~Completer() override;

    CallbackReturn SearchCallback(SearchFilter &filter, SymbolContext &context,
                                  Address *addr, bool complete) override = 0;

    Depth GetDepth() override = 0;

    virtual size_t DoCompletion(SearchFilter *filter) = 0;

  protected:
    CommandInterpreter &m_interpreter;
    std::string m_completion_str;
    int m_match_start_point;
    int m_max_return_elements;
    StringList &m_matches;

  private:
    DISALLOW_COPY_AND_ASSIGN(Completer);
  };

  //----------------------------------------------------------------------
  // SourceFileCompleter implements the source file completer
  //----------------------------------------------------------------------
  class SourceFileCompleter : public Completer {
  public:
    SourceFileCompleter(CommandInterpreter &interpreter,
                        bool include_support_files,
                        llvm::StringRef completion_str, int match_start_point,
                        int max_return_elements, StringList &matches);

    Searcher::Depth GetDepth() override;

    Searcher::CallbackReturn SearchCallback(SearchFilter &filter,
                                            SymbolContext &context,
                                            Address *addr,
                                            bool complete) override;

    size_t DoCompletion(SearchFilter *filter) override;

  private:
    bool m_include_support_files;
    FileSpecList m_matching_files;
    const char *m_file_name;
    const char *m_dir_name;

    DISALLOW_COPY_AND_ASSIGN(SourceFileCompleter);
  };

  //----------------------------------------------------------------------
  // ModuleCompleter implements the module completer
  //----------------------------------------------------------------------
  class ModuleCompleter : public Completer {
  public:
    ModuleCompleter(CommandInterpreter &interpreter,
                    llvm::StringRef completion_str, int match_start_point,
                    int max_return_elements, StringList &matches);

    Searcher::Depth GetDepth() override;

    Searcher::CallbackReturn SearchCallback(SearchFilter &filter,
                                            SymbolContext &context,
                                            Address *addr,
                                            bool complete) override;

    size_t DoCompletion(SearchFilter *filter) override;

  private:
    const char *m_file_name;
    const char *m_dir_name;

    DISALLOW_COPY_AND_ASSIGN(ModuleCompleter);
  };

  //----------------------------------------------------------------------
  // SymbolCompleter implements the symbol completer
  //----------------------------------------------------------------------
  class SymbolCompleter : public Completer {
  public:
    SymbolCompleter(CommandInterpreter &interpreter,
                    llvm::StringRef completion_str, int match_start_point,
                    int max_return_elements, StringList &matches);

    Searcher::Depth GetDepth() override;

    Searcher::CallbackReturn SearchCallback(SearchFilter &filter,
                                            SymbolContext &context,
                                            Address *addr,
                                            bool complete) override;

    size_t DoCompletion(SearchFilter *filter) override;

  private:
    //        struct NameCmp {
    //            bool operator() (const ConstString& lhs, const ConstString&
    //            rhs) const
    //            {
    //                return lhs < rhs;
    //            }
    //        };

    RegularExpression m_regex;
    typedef std::set<ConstString> collection;
    collection m_match_set;

    DISALLOW_COPY_AND_ASSIGN(SymbolCompleter);
  };

private:
  static CommonCompletionElement g_common_completions[];
};

} // namespace lldb_private

#endif // lldb_CommandCompletions_h_
