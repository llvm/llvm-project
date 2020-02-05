//===-- StringPrinter.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StringPrinter_h_
#define liblldb_StringPrinter_h_

#include <functional>
#include <string>

#include "lldb/lldb-forward.h"

#include "lldb/Utility/DataExtractor.h"

namespace lldb_private {
namespace formatters {
class StringPrinter {
public:
  enum class StringElementType { ASCII, UTF8, UTF16, UTF32 };

  enum class GetPrintableElementType { ASCII, UTF8 };

  class DumpToStreamOptions {
  public:
    DumpToStreamOptions() = default;

    void SetStream(Stream *s) { m_stream = s; }

    Stream *GetStream() const { return m_stream; }

    void SetPrefixToken(const std::string &p) { m_prefix_token = p; }

    void SetPrefixToken(std::nullptr_t) { m_prefix_token.clear(); }

    const char *GetPrefixToken() const { return m_prefix_token.c_str(); }

    void SetSuffixToken(const std::string &p) { m_suffix_token = p; }

    void SetSuffixToken(std::nullptr_t) { m_suffix_token.clear(); }

    const char *GetSuffixToken() const { return m_suffix_token.c_str(); }

    void SetQuote(char q) { m_quote = q; }

    char GetQuote() const { return m_quote; }

    void SetSourceSize(uint32_t s) { m_source_size = s; }

    uint32_t GetSourceSize() const { return m_source_size; }

    void SetNeedsZeroTermination(bool z) { m_needs_zero_termination = z; }

    bool GetNeedsZeroTermination() const { return m_needs_zero_termination; }

    void SetBinaryZeroIsTerminator(bool e) { m_zero_is_terminator = e; }

    bool GetBinaryZeroIsTerminator() const { return m_zero_is_terminator; }

    void SetEscapeNonPrintables(bool e) { m_escape_non_printables = e; }

    bool GetEscapeNonPrintables() const { return m_escape_non_printables; }

    void SetIgnoreMaxLength(bool e) { m_ignore_max_length = e; }

    bool GetIgnoreMaxLength() const { return m_ignore_max_length; }

    void SetLanguage(lldb::LanguageType l) { m_language_type = l; }

    lldb::LanguageType GetLanguage() const { return m_language_type; }

  private:
    /// The used output stream.
    Stream *m_stream = nullptr;
    /// String that should be printed before the heading quote character.
    std::string m_prefix_token;
    /// String that should be printed after the trailing quote character.
    std::string m_suffix_token;
    /// The quote character that should surround the string.
    char m_quote = '"';
    /// The length of the memory region that should be dumped in bytes.
    uint32_t m_source_size = 0;
    bool m_needs_zero_termination = true;
    /// True iff non-printable characters should be escaped when dumping
    /// them to the stream.
    bool m_escape_non_printables = true;
    /// True iff the max-string-summary-length setting of the target should
    /// be ignored.
    bool m_ignore_max_length = false;
    /// True iff a zero bytes ('\0') should terminate the memory region that
    /// is being dumped.
    bool m_zero_is_terminator = true;
    /// The language that the generated string literal is supposed to be valid
    /// for. This changes for example what and how certain characters are
    /// escaped.
    /// For example, printing the a string containing only a quote (") char
    /// with eLanguageTypeC would escape the quote character.
    lldb::LanguageType m_language_type = lldb::eLanguageTypeUnknown;
  };

  class ReadStringAndDumpToStreamOptions : public DumpToStreamOptions {
  public:
    ReadStringAndDumpToStreamOptions() = default;

    ReadStringAndDumpToStreamOptions(ValueObject &valobj);

    void SetLocation(uint64_t l) { m_location = l; }

    uint64_t GetLocation() const { return m_location; }

    void SetProcessSP(lldb::ProcessSP p) { m_process_sp = p; }

    lldb::ProcessSP GetProcessSP() const { return m_process_sp; }

  private:
    uint64_t m_location = 0;
    lldb::ProcessSP m_process_sp;
  };

  class ReadBufferAndDumpToStreamOptions : public DumpToStreamOptions {
  public:
    ReadBufferAndDumpToStreamOptions() = default;

    ReadBufferAndDumpToStreamOptions(ValueObject &valobj);

    ReadBufferAndDumpToStreamOptions(
        const ReadStringAndDumpToStreamOptions &options);

    void SetData(DataExtractor d) { m_data = d; }

    lldb_private::DataExtractor GetData() const { return m_data; }

    void SetIsTruncated(bool t) { m_is_truncated = t; }

    bool GetIsTruncated() const { return m_is_truncated; }
  private:
    DataExtractor m_data;
    bool m_is_truncated = false;
  };

  // I can't use a std::unique_ptr for this because the Deleter is a template
  // argument there
  // and I want the same type to represent both pointers I want to free and
  // pointers I don't need to free - which is what this class essentially is
  // It's very specialized to the needs of this file, and not suggested for
  // general use
  struct StringPrinterBufferPointer {
  public:
    typedef std::function<void(const uint8_t *)> Deleter;

    StringPrinterBufferPointer(std::nullptr_t ptr)
        : m_data(nullptr), m_size(0), m_deleter() {}

    StringPrinterBufferPointer(const uint8_t *bytes, size_t size,
                               Deleter deleter = nullptr)
        : m_data(bytes), m_size(size), m_deleter(deleter) {}

    StringPrinterBufferPointer(const char *bytes, size_t size,
                               Deleter deleter = nullptr)
        : m_data(reinterpret_cast<const uint8_t *>(bytes)), m_size(size),
          m_deleter(deleter) {}

    StringPrinterBufferPointer(StringPrinterBufferPointer &&rhs)
        : m_data(rhs.m_data), m_size(rhs.m_size), m_deleter(rhs.m_deleter) {
      rhs.m_data = nullptr;
    }

    ~StringPrinterBufferPointer() {
      if (m_data && m_deleter)
        m_deleter(m_data);
      m_data = nullptr;
    }

    const uint8_t *GetBytes() const { return m_data; }

    size_t GetSize() const { return m_size; }

    StringPrinterBufferPointer &
    operator=(StringPrinterBufferPointer &&rhs) {
      if (m_data && m_deleter)
        m_deleter(m_data);
      m_data = rhs.m_data;
      m_size = rhs.m_size;
      m_deleter = rhs.m_deleter;
      rhs.m_data = nullptr;
      return *this;
    }

  private:
    DISALLOW_COPY_AND_ASSIGN(StringPrinterBufferPointer);

    const uint8_t *m_data;
    size_t m_size;
    Deleter m_deleter;
  };

  typedef std::function<StringPrinter::StringPrinterBufferPointer(
      uint8_t *, uint8_t *, uint8_t *&)>
      EscapingHelper;
  typedef std::function<EscapingHelper(GetPrintableElementType)>
      EscapingHelperGenerator;

  static EscapingHelper
  GetDefaultEscapingHelper(GetPrintableElementType elem_type);

  template <StringElementType element_type>
  static bool
  ReadStringAndDumpToStream(const ReadStringAndDumpToStreamOptions &options);

  template <StringElementType element_type>
  static bool
  ReadBufferAndDumpToStream(const ReadBufferAndDumpToStreamOptions &options);
};

} // namespace formatters
} // namespace lldb_private

#endif // liblldb_StringPrinter_h_
