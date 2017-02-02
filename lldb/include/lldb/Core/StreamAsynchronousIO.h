//===-- StreamAsynchronousIO.h -----------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamAsynchronousIO_h_
#define liblldb_StreamAsynchronousIO_h_

#include <string>

#include "lldb/Utility/Stream.h"

namespace lldb_private {

class StreamAsynchronousIO : public Stream {
public:
  StreamAsynchronousIO(Debugger &debugger, bool for_stdout);

  ~StreamAsynchronousIO() override;

  void Flush() override;

  size_t Write(const void *src, size_t src_len) override;

private:
  Debugger &m_debugger;
  std::string m_data;
  bool m_for_stdout;
};

} // namespace lldb_private

#endif // liblldb_StreamAsynchronousIO_h
