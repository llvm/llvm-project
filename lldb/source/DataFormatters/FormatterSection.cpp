//===-- FormatterBytecode.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatterBytecode.h"
#include "lldb/Core/Module.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/Utility/LLDBLog.h"

using namespace lldb;

namespace lldb_private {
static void ForEachFormatterInModule(
    Module &module, SectionType section_type,
    std::function<void(llvm::DataExtractor, llvm::StringRef)> fn) {
  auto *sections = module.GetSectionList();
  if (!sections)
    return;

  auto section_sp = sections->FindSectionByType(section_type, true);
  if (!section_sp)
    return;

  TypeCategoryImplSP category;
  DataVisualization::Categories::GetCategory(ConstString("default"), category);

  // The type summary record is serialized as follows.
  //
  // Each record contains, in order:
  //   * Version number of the record format
  //   * The remaining size of the record
  //   * The size of the type identifier
  //   * The type identifier, either a type name, or a regex
  //   * The size of the entry
  //   * The entry
  //
  // Integers are encoded using ULEB.
  //
  // Strings are encoded with first a length (ULEB), then the string contents,
  // and lastly a null terminator. The length includes the null.

  DataExtractor lldb_extractor;
  auto section_size = section_sp->GetSectionData(lldb_extractor);
  llvm::DataExtractor section = lldb_extractor.GetAsLLVM();
  bool le = section.isLittleEndian();
  uint8_t addr_size = section.getAddressSize();
  llvm::DataExtractor::Cursor cursor(0);
  while (cursor && cursor.tell() < section_size) {
    while (cursor && cursor.tell() < section_size) {
      // Skip over 0 padding.
      if (section.getU8(cursor) == 0)
        continue;
      cursor.seek(cursor.tell() - 1);
      break;
    }
    uint64_t version = section.getULEB128(cursor);
    uint64_t record_size = section.getULEB128(cursor);
    if (version == 1) {
      llvm::DataExtractor record(section.getData().drop_front(cursor.tell()),
                                 le, addr_size);
      llvm::DataExtractor::Cursor cursor(0);
      uint64_t type_size = record.getULEB128(cursor);
      llvm::StringRef type_name = record.getBytes(cursor, type_size);
      llvm::Error error = cursor.takeError();
      if (!error)
        fn(llvm::DataExtractor(record.getData().drop_front(cursor.tell()), le,
                               addr_size),
           type_name);
      else
        LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters), std::move(error),
                       "{0}");
    } else {
      // Skip unsupported record.
      LLDB_LOG(
          GetLog(LLDBLog::DataFormatters),
          "Skipping unsupported embedded type summary of version {0} in {1}.",
          version, module.GetFileSpec());
    }
    section.skip(cursor, record_size);
  }
  if (!cursor)
    LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters), cursor.takeError(), "{0}");
}

void LoadTypeSummariesForModule(ModuleSP module_sp) {
  ForEachFormatterInModule(
      *module_sp, eSectionTypeLLDBTypeSummaries,
      [&](llvm::DataExtractor extractor, llvm::StringRef type_name) {
        TypeCategoryImplSP category;
        DataVisualization::Categories::GetCategory(ConstString("default"),
                                                   category);
        // The type summary record is serialized as follows.
        //
        //   * The size of the summary string
        //   * The summary string
        //
        // Integers are encoded using ULEB.
        llvm::DataExtractor::Cursor cursor(0);
        uint64_t summary_size = extractor.getULEB128(cursor);
        llvm::StringRef summary_string =
            extractor.getBytes(cursor, summary_size);
        if (!cursor) {
          LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters), cursor.takeError(),
                         "{0}");
          return;
        }
        if (type_name.empty() || summary_string.empty()) {
          LLDB_LOG(GetLog(LLDBLog::DataFormatters),
                   "Missing string(s) in embedded type summary in {0}, "
                   "type_name={1}, summary={2}",
                   module_sp->GetFileSpec(), type_name, summary_string);
          return;
        }
        TypeSummaryImpl::Flags flags;
        auto summary_sp = std::make_shared<StringSummaryFormat>(
            flags, summary_string.str().c_str());
        FormatterMatchType match_type = eFormatterMatchExact;
        if (type_name.front() == '^')
          match_type = eFormatterMatchRegex;
        category->AddTypeSummary(type_name, match_type, summary_sp);
        LLDB_LOG(GetLog(LLDBLog::DataFormatters),
                 "Loaded embedded type summary for '{0}' from {1}.", type_name,
                 module_sp->GetFileSpec());
      });
}

void LoadFormattersForModule(ModuleSP module_sp) {
  ForEachFormatterInModule(
      *module_sp, eSectionTypeLLDBFormatters,
      [&](llvm::DataExtractor extractor, llvm::StringRef type_name) {
        // * Function signature (1 byte)
        // * Length of the program (ULEB128)
        // * The program bytecode
        TypeCategoryImplSP category;
        DataVisualization::Categories::GetCategory(ConstString("default"),
                                                   category);
        llvm::DataExtractor::Cursor cursor(0);
        uint64_t flags = extractor.getULEB128(cursor);
        while (cursor && cursor.tell() < extractor.size()) {
          uint8_t signature = extractor.getU8(cursor);
          uint64_t size = extractor.getULEB128(cursor);
          llvm::StringRef bytecode = extractor.getBytes(cursor, size);
          if (!cursor) {
            LLDB_LOG_ERROR(GetLog(LLDBLog::DataFormatters), cursor.takeError(),
                           "{0}");
            return;
          }
          if (signature == 0) {
            auto summary_sp = std::make_shared<BytecodeSummaryFormat>(
                TypeSummaryImpl::Flags(flags),
                llvm::MemoryBuffer::getMemBufferCopy(bytecode));
            FormatterMatchType match_type = eFormatterMatchExact;
            if (type_name.front() == '^')
              match_type = eFormatterMatchRegex;
            category->AddTypeSummary(type_name, match_type, summary_sp);
            LLDB_LOG(GetLog(LLDBLog::DataFormatters),
                     "Loaded embedded type summary for '{0}' from {1}.",
                     type_name, module_sp->GetFileSpec());
          } else
            LLDB_LOG(GetLog(LLDBLog::DataFormatters),
                     "Unsupported formatter signature {0} for '{1}' in {2}",
                     signature, type_name, module_sp->GetFileSpec());
        }
      });
}
} // namespace lldb_private
