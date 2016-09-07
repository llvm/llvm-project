//===-- StreamGDBRemote.h ----------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamGDBRemote_h_
#define liblldb_StreamGDBRemote_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes

#include "lldb/Core/StreamString.h"

namespace lldb_private {

class StreamGDBRemote : public StreamString {
public:
  StreamGDBRemote();

  StreamGDBRemote(uint32_t flags, uint32_t addr_size,
                  lldb::ByteOrder byte_order);

  ~StreamGDBRemote() override;

  //------------------------------------------------------------------
  /// Output a block of data to the stream performing GDB-remote escaping.
  ///
  /// @param[in] s
  ///     A block of data.
  ///
  /// @param[in] src_len
  ///     The amount of data to write.
  ///
  /// @return
  ///     Number of bytes written.
  //------------------------------------------------------------------
  int PutEscapedBytes(const void *s, size_t src_len);
};

} // namespace lldb_private

#endif // liblldb_StreamGDBRemote_h_
