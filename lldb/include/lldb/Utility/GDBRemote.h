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
#include "lldb/Utility/ReproducerProvider.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-public.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
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

namespace repro {
class PacketRecorder : public AbstractRecorder {
public:
  PacketRecorder(const FileSpec &filename, std::error_code &ec)
      : AbstractRecorder(filename, ec) {}

  static llvm::Expected<std::unique_ptr<PacketRecorder>>
  Create(const FileSpec &filename);

  void Record(const GDBRemotePacket &packet);
};

class GDBRemoteProvider : public repro::Provider<GDBRemoteProvider> {
public:
  struct Info {
    static const char *name;
    static const char *file;
  };

  GDBRemoteProvider(const FileSpec &directory) : Provider(directory) {}

  llvm::raw_ostream *GetHistoryStream();
  PacketRecorder *GetNewPacketRecorder();

  void SetCallback(std::function<void()> callback) {
    m_callback = std::move(callback);
  }

  void Keep() override;
  void Discard() override;

  static char ID;

private:
  std::function<void()> m_callback;
  std::unique_ptr<llvm::raw_fd_ostream> m_stream_up;
  std::vector<std::unique_ptr<PacketRecorder>> m_packet_recorders;
};

} // namespace repro
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

#endif // LLDB_UTILITY_GDBREMOTE_H
