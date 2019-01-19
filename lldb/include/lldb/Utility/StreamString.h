//===-- StreamString.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamString_h_
#define liblldb_StreamString_h_

#include "lldb/Utility/Stream.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/StringRef.h"

#include <string>

#include <stddef.h>
#include <stdint.h>

namespace lldb_private {

class StreamString : public Stream {
public:
  StreamString();

  StreamString(uint32_t flags, uint32_t addr_size, lldb::ByteOrder byte_order);

  ~StreamString() override;

  void Flush() override;

  void Clear();

  bool Empty() const;

  size_t GetSize() const;

  size_t GetSizeOfLastLine() const;

  llvm::StringRef GetString() const;

  const char *GetData() const { return m_packet.c_str(); }

  void FillLastLineToColumn(uint32_t column, char fill_char);

protected:
  std::string m_packet;
  size_t WriteImpl(const void *s, size_t length) override;
};

} // namespace lldb_private

#endif // liblldb_StreamString_h_
