//===-- CommandCompletions.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef lldb_CommandCompletions_h_
#define lldb_CommandCompletions_h_

#include <set>

#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/SearchFilter.h"
#include "lldb/Utility/CompletionRequest.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/lldb-private.h"

#include "llvm/ADT/Twine.h"

namespace lldb_private {
class TildeExpressionResolver;
class CommandCompletions {
public:
  // This is the command completion callback that is used to complete the
  // argument of the option it is bound to (in the OptionDefinition table
  // below).  Return the total number of matches.
  typedef void (*CompletionCallback)(CommandInterpreter &interpreter,
                                     CompletionRequest &request,
                                     // A search filter to limit the search...
                                     lldb_private::SearchFilter *searcher);
  enum CommonCompletionTypes {
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
    // This item serves two purposes.  It is the last element in the enum, so
    // you can add custom enums starting from here in your Option class. Also
    // if you & in this bit the base code will not process the option.
    eCustomCompletion = (1u << 9)
  };

  struct CommonCompletionElement {
    uint32_t type;
    CompletionCallback callback;
  };

  static bool InvokeCommonCompletionCallbacks(
      CommandInterpreter &interpreter, uint32_t completion_mask,
      lldb_private::CompletionRequest &request, SearchFilter *searcher);

  // These are the generic completer functions:
  static void DiskFiles(CommandInterpreter &interpreter,
                        CompletionRequest &request, SearchFilter *searcher);

  static void DiskFiles(const llvm::Twine &partial_file_name,
                        StringList &matches, TildeExpressionResolver &Resolver);

  static void DiskDirectories(CommandInterpreter &interpreter,
                              CompletionRequest &request,
                              SearchFilter *searcher);

  static void DiskDirectories(const llvm::Twine &partial_file_name,
                              StringList &matches,
                              TildeExpressionResolver &Resolver);

  static void SourceFiles(CommandInterpreter &interpreter,
                          CompletionRequest &request, SearchFilter *searcher);

  static void Modules(CommandInterpreter &interpreter,
                      CompletionRequest &request, SearchFilter *searcher);

  static void Symbols(CommandInterpreter &interpreter,
                      CompletionRequest &request, SearchFilter *searcher);

  static void SettingsNames(CommandInterpreter &interpreter,
                            CompletionRequest &request, SearchFilter *searcher);

  static void PlatformPluginNames(CommandInterpreter &interpreter,
                                  CompletionRequest &request,
                                  SearchFilter *searcher);

  static void ArchitectureNames(CommandInterpreter &interpreter,
                                CompletionRequest &request,
                                SearchFilter *searcher);

  static void VariablePath(CommandInterpreter &interpreter,
                           CompletionRequest &request, SearchFilter *searcher);

private:
  static CommonCompletionElement g_common_completions[];
};

} // namespace lldb_private

#endif // lldb_CommandCompletions_h_
