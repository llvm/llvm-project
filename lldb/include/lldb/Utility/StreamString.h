//===-- StreamString.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamString_h_
#define liblldb_StreamString_h_

#include "lldb/Utility/Stream.h"    // for Stream
#include "lldb/lldb-enumerations.h" // for ByteOrder
#include "llvm/ADT/StringRef.h"     // for StringRef

#include <string> // for string

#include <stddef.h> // for size_t
#include <stdint.h> // for uint32_t

namespace lldb_private {

class StreamString : public Stream {
public:
  StreamString();

  StreamString(uint32_t flags, uint32_t addr_size, lldb::ByteOrder byte_order);

  ~StreamString() override;

  void Flush() override;

  size_t Write(const void *s, size_t length) override;

  void Clear();

  bool Empty() const;

  size_t GetSize() const;

  size_t GetSizeOfLastLine() const;

  llvm::StringRef GetString() const;

  const char *GetData() const { return m_packet.c_str(); }

  void FillLastLineToColumn(uint32_t column, char fill_char);

protected:
  std::string m_packet;
};

} // namespace lldb_private

#endif // liblldb_StreamString_h_
