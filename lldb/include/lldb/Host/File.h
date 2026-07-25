//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_FILE_H
#define LLDB_HOST_FILE_H

#include "lldb/Host/FileBase.h"
#include "lldb/Host/Terminal.h"

#include <memory>
#include <optional>

#if defined(_WIN32)
#include "lldb/Host/windows/FileWindows.h"
#else
#include "lldb/Host/posix/FilePosix.h"
#endif

namespace lldb_private {

#if defined(_WIN32)
typedef NativeFileWindows NativeFile;
#else
typedef NativeFilePosix NativeFile;
#endif

class SerialPort : public NativeFile {
public:
  struct Options {
    std::optional<unsigned int> BaudRate;
    std::optional<Terminal::Parity> Parity;
    std::optional<Terminal::ParityCheck> ParityCheck;
    std::optional<unsigned int> StopBits;
  };

  // Obtain Options corresponding to the passed URL query string
  // (i.e. the part after '?').
  static llvm::Expected<Options> OptionsFromURL(llvm::StringRef urlqs);

  static llvm::Expected<std::unique_ptr<SerialPort>>
  Create(int fd, OpenOptions options, Options serial_options,
         bool transfer_ownership);

  bool IsValid() const override {
    return NativeFile::IsValid() && m_is_interactive == eLazyBoolYes;
  }

  Status Close() override;

  static char ID;
  bool isA(const void *classID) const override {
    return classID == &ID || NativeFile::isA(classID);
  }
  static bool classof(const File *file) { return file->isA(&ID); }

private:
  SerialPort(int fd, OpenOptions options, Options serial_options,
             bool transfer_ownership);

  SerialPort(const SerialPort &) = delete;
  const SerialPort &operator=(const SerialPort &) = delete;

  TerminalState m_state;
};

} // namespace lldb_private

#endif // LLDB_HOST_FILE_H
