//===-- GDBRemote.h ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_GDBREMOTE_H
#define LLDB_UTILITY_GDBREMOTE_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-public.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lldb_private {

class StreamGDBRemote : public StreamString {
public:
  StreamGDBRemote();

  StreamGDBRemote(uint32_t flags, lldb::ByteOrder byte_order);

  ~StreamGDBRemote() override;

  /// Output a block of data to the stream performing GDB-remote escaping.
  ///
  /// \param[in] bytes
  ///     A block of data.
  ///
  /// \return
  ///     Number of bytes written.
  int PutEscapedBytes(llvm::ArrayRef<uint8_t> bytes);

  /// \overload
  /// This overload is provided for backward compatibility with the existing
  /// code. The newer interface is to use the ArrayRef<uint8_t> overload.
  int PutEscapedBytes(const void *src, size_t len) {
    return PutEscapedBytes(
        llvm::ArrayRef<uint8_t>(static_cast<const uint8_t *>(src), len));
  }

  template <class T> int PutAsJSON(const T &obj, bool hex_ascii) {
    std::string json_string;
    llvm::raw_string_ostream os(json_string);
    os << llvm::json::Value(toJSON(obj));
    if (hex_ascii)
      return PutStringAsRawHex8(json_string);
    return PutEscapedBytes(llvm::arrayRefFromStringRef(json_string));
  }

  template <class T>
  int PutAsJSONArray(llvm::ArrayRef<T> array, bool hex_ascii) {
    llvm::json::Array json_array;
    for (const auto &obj : array)
      json_array.push_back(toJSON(obj));
    std::string json_string;
    llvm::raw_string_ostream os(json_string);
    os << llvm::json::Value(std::move(json_array));
    if (hex_ascii)
      return PutStringAsRawHex8(json_string);
    return PutEscapedBytes(llvm::arrayRefFromStringRef(json_string));
  }
};

/// GDB remote packet as used by the GDB remote communication history. Packets
/// can be serialized to file.
struct GDBRemotePacket {

  enum Type { ePacketTypeInvalid = 0, ePacketTypeSend, ePacketTypeRecv };

  GDBRemotePacket() = default;

  void Clear() {
    packet.data.clear();
    type = ePacketTypeInvalid;
    bytes_transmitted = 0;
    packet_idx = 0;
    tid = LLDB_INVALID_THREAD_ID;
  }

  struct BinaryData {
    std::string data;
  };

  void Dump(Stream &strm) const;

  BinaryData packet;
  Type type = ePacketTypeInvalid;
  uint32_t bytes_transmitted = 0;
  uint32_t packet_idx = 0;
  lldb::tid_t tid = LLDB_INVALID_THREAD_ID;

private:
  llvm::StringRef GetTypeStr() const;
};

} // namespace lldb_private

#endif // LLDB_UTILITY_GDBREMOTE_H
