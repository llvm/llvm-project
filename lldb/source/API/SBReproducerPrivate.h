//===-- SBReproducerPrivate.h -----------------------------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_API_SBREPRODUCERPRIVATE_H
#define LLDB_SOURCE_API_SBREPRODUCERPRIVATE_H

#include "lldb/API/SBReproducer.h"

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Reproducer.h"
#include "lldb/Utility/ReproducerInstrumentation.h"

#include "llvm/ADT/DenseMap.h"

#define LLDB_GET_INSTRUMENTATION_DATA()                                        \
  lldb_private::repro::InstrumentationData::Instance()

namespace lldb_private {
namespace repro {

class SBRegistry : public Registry {
public:
  SBRegistry();
};

class SBProvider : public Provider<SBProvider> {
public:
  struct Info {
    static const char *name;
    static const char *file;
  };

  SBProvider(const FileSpec &directory)
      : Provider(directory),
        m_stream(directory.CopyByAppendingPathComponent("sbapi.bin").GetPath(),
                 m_ec, llvm::sys::fs::OpenFlags::OF_None),
        m_serializer(m_stream) {}

  Serializer &GetSerializer() { return m_serializer; }
  Registry &GetRegistry() { return m_registry; }

  static char ID;

private:
  std::error_code m_ec;
  llvm::raw_fd_ostream m_stream;
  Serializer m_serializer;
  SBRegistry m_registry;
};

class ReplayData {
public:
  ReplayData(std::unique_ptr<llvm::MemoryBuffer> memory_buffer)
      : m_memory_buffer(std::move(memory_buffer)), m_registry(),
        m_deserializer(m_memory_buffer->getBuffer()) {}
  Deserializer &GetDeserializer() { return m_deserializer; }
  Registry &GetRegistry() { return m_registry; }

private:
  std::unique_ptr<llvm::MemoryBuffer> m_memory_buffer;
  SBRegistry m_registry;
  Deserializer m_deserializer;
};

template <typename T> void RegisterMethods(Registry &R);

} // namespace repro
} // namespace lldb_private

#endif
