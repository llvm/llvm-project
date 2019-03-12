//===-- StreamGDBRemote.h ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamGDBRemote_h_
#define liblldb_StreamGDBRemote_h_

#include "lldb/Utility/StreamString.h"
#include "lldb/lldb-enumerations.h"

#include <stddef.h>
#include <stdint.h>

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
  /// \param[in] s
  ///     A block of data.
  ///
  /// \param[in] src_len
  ///     The amount of data to write.
  ///
  /// \return
  ///     Number of bytes written.
  //------------------------------------------------------------------
  // TODO: Convert this function to take ArrayRef<uint8_t>
  int PutEscapedBytes(const void *s, size_t src_len);
};

} // namespace lldb_private

#endif // liblldb_StreamGDBRemote_h_
