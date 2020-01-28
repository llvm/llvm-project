//===-- FormatEntity.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_FormatEntity_h_
#define liblldb_FormatEntity_h_

#include "lldb/Utility/CompletionRequest.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-types.h"
#include <algorithm>
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

namespace lldb_private {
class Address;
class ExecutionContext;
class Stream;
class StringList;
class SymbolContext;
class ValueObject;
}
namespace llvm {
class StringRef;
}

namespace lldb_private {
class FormatEntity {
public:
  struct Entry {
    enum class Type {
      Invalid,
      ParentNumber,
      ParentString,
      EscapeCode,
      Root,
      String,
      Scope,
      Variable,
      VariableSynthetic,
      ScriptVariable,
      ScriptVariableSynthetic,
      AddressLoad,
      AddressFile,
      AddressLoadOrFile,
      ProcessID,
      ProcessFile,
      ScriptProcess,
      ThreadID,
      ThreadProtocolID,
      ThreadIndexID,
      ThreadName,
      ThreadQueue,
      ThreadStopReason,
      ThreadReturnValue,
      ThreadCompletedExpression,
      ScriptThread,
      ThreadInfo,
      TargetArch,
      ScriptTarget,
      ModuleFile,
      File,
      Lang,
      FrameIndex,
      FrameNoDebug,
      FrameRegisterPC,
      FrameRegisterSP,
      FrameRegisterFP,
      FrameRegisterFlags,
      FrameRegisterByName,
      FrameIsArtificial,
      ScriptFrame,
      FunctionID,
      FunctionDidChange,
      FunctionInitialFunction,
      FunctionName,
      FunctionNameWithArgs,
      FunctionNameNoArgs,
      FunctionMangledName,
      FunctionAddrOffset,
      FunctionAddrOffsetConcrete,
      FunctionLineOffset,
      FunctionPCOffset,
      FunctionInitial,
      FunctionChanged,
      FunctionIsOptimized,
      LineEntryFile,
      LineEntryLineNumber,
      LineEntryColumn,
      LineEntryStartAddress,
      LineEntryEndAddress,
      CurrentPCArrow
    };

    struct Definition {
      const char *name;
      const char *string; // Insert this exact string into the output
      Entry::Type type;
      uint64_t data;
      uint32_t num_children;
      Definition *children; // An array of "num_children" Definition entries,
      bool keep_separator;
    };

    Entry(Type t = Type::Invalid, const char *s = nullptr,
          const char *f = nullptr)
        : string(s ? s : ""), printf_format(f ? f : ""), children(),
          definition(nullptr), type(t), fmt(lldb::eFormatDefault), number(0),
          deref(false) {}

    Entry(llvm::StringRef s);
    Entry(char ch);

    void AppendChar(char ch);

    void AppendText(const llvm::StringRef &s);

    void AppendText(const char *cstr);

    void AppendEntry(const Entry &&entry) { children.push_back(entry); }

    void Clear() {
      string.clear();
      printf_format.clear();
      children.clear();
      definition = nullptr;
      type = Type::Invalid;
      fmt = lldb::eFormatDefault;
      number = 0;
      deref = false;
    }

    static const char *TypeToCString(Type t);

    void Dump(Stream &s, int depth = 0) const;

    bool operator==(const Entry &rhs) const {
      if (string != rhs.string)
        return false;
      if (printf_format != rhs.printf_format)
        return false;
      const size_t n = children.size();
      const size_t m = rhs.children.size();
      for (size_t i = 0; i < std::min<size_t>(n, m); ++i) {
        if (!(children[i] == rhs.children[i]))
          return false;
      }
      if (children != rhs.children)
        return false;
      if (definition != rhs.definition)
        return false;
      if (type != rhs.type)
        return false;
      if (fmt != rhs.fmt)
        return false;
      if (deref != rhs.deref)
        return false;
      return true;
    }

    std::string string;
    std::string printf_format;
    std::vector<Entry> children;
    Definition *definition;
    Type type;
    lldb::Format fmt;
    lldb::addr_t number;
    bool deref;
  };

  static bool Format(const Entry &entry, Stream &s, const SymbolContext *sc,
                     const ExecutionContext *exe_ctx, const Address *addr,
                     ValueObject *valobj, bool function_changed,
                     bool initial_function);

  static bool FormatStringRef(const llvm::StringRef &format, Stream &s,
                              const SymbolContext *sc,
                              const ExecutionContext *exe_ctx,
                              const Address *addr, ValueObject *valobj,
                              bool function_changed, bool initial_function);

  static bool FormatCString(const char *format, Stream &s,
                            const SymbolContext *sc,
                            const ExecutionContext *exe_ctx,
                            const Address *addr, ValueObject *valobj,
                            bool function_changed, bool initial_function);

  static Status Parse(const llvm::StringRef &format, Entry &entry);

  static Status ExtractVariableInfo(llvm::StringRef &format_str,
                                    llvm::StringRef &variable_name,
                                    llvm::StringRef &variable_format);

  static void AutoComplete(lldb_private::CompletionRequest &request);

  // Format the current elements into the stream \a s.
  //
  // The root element will be stripped off and the format str passed in will be
  // either an empty string (print a description of this object), or contain a
  // `.`-separated series like a domain name that identifies further
  //  sub-elements to display.
  static bool FormatFileSpec(const FileSpec &file, Stream &s,
                             llvm::StringRef elements,
                             llvm::StringRef element_format);

protected:
  static Status ParseInternal(llvm::StringRef &format, Entry &parent_entry,
                              uint32_t depth);
};
} // namespace lldb_private

#endif // liblldb_FormatEntity_h_
