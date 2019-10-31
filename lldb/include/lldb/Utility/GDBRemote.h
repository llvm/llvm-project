//===-- GDBRemote.h ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GDBRemote_h_
#define liblldb_GDBRemote_h_

#include "lldb/Utility/StreamString.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-public.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace lldb_private {

class StreamGDBRemote : public StreamString {
public:
  StreamGDBRemote();

  StreamGDBRemote(uint32_t flags, uint32_t addr_size,
                  lldb::ByteOrder byte_order);

  ~StreamGDBRemote() override;

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
  // TODO: Convert this function to take ArrayRef<uint8_t>
  int PutEscapedBytes(const void *s, size_t src_len);
};

/// GDB remote packet as used by the reproducer and the GDB remote
/// communication history. Packets can be serialized to file.
struct GDBRemotePacket {

  friend llvm::yaml::MappingTraits<GDBRemotePacket>;

  enum Type { ePacketTypeInvalid = 0, ePacketTypeSend, ePacketTypeRecv };

  GDBRemotePacket()
      : packet(), type(ePacketTypeInvalid), bytes_transmitted(0), packet_idx(0),
        tid(LLDB_INVALID_THREAD_ID) {}

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

  void Serialize(llvm::raw_ostream &strm) const;
  void Dump(Stream &strm) const;

  BinaryData packet;
  Type type;
  uint32_t bytes_transmitted;
  uint32_t packet_idx;
  lldb::tid_t tid;

private:
  llvm::StringRef GetTypeStr() const;
};

} // namespace lldb_private

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(lldb_private::GDBRemotePacket)
LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(std::vector<lldb_private::GDBRemotePacket>)

namespace llvm {
namespace yaml {

template <>
struct ScalarEnumerationTraits<lldb_private::GDBRemotePacket::Type> {
  static void enumeration(IO &io, lldb_private::GDBRemotePacket::Type &value);
};

template <> struct ScalarTraits<lldb_private::GDBRemotePacket::BinaryData> {
  static void output(const lldb_private::GDBRemotePacket::BinaryData &, void *,
                     raw_ostream &);

  static StringRef input(StringRef, void *,
                         lldb_private::GDBRemotePacket::BinaryData &);

  static QuotingType mustQuote(StringRef S) { return QuotingType::None; }
};

template <> struct MappingTraits<lldb_private::GDBRemotePacket> {
  static void mapping(IO &io, lldb_private::GDBRemotePacket &Packet);

  static StringRef validate(IO &io, lldb_private::GDBRemotePacket &);
};

} // namespace yaml
} // namespace llvm

#endif // liblldb_GDBRemote_h_
