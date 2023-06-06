//===--- GPU specialization of the File data structure --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/File/file.h"

#include "src/__support/RPC/rpc_client.h"
#include "src/errno/libc_errno.h" // For error macros

#include <stdio.h>

namespace __llvm_libc {

namespace {

FileIOResult write_func(File *, const void *, size_t);

} // namespace

class GPUFile : public File {
  uintptr_t file;

public:
  constexpr GPUFile(uintptr_t file, File::ModeFlags modeflags)
      : File(&write_func, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
             0, _IONBF, false, modeflags),
        file(file) {}

  uintptr_t get_file() const { return file; }
};

namespace {

int write_to_stdout(const void *data, size_t size) {
  int ret = 0;
  rpc::Client::Port port = rpc::client.open<rpc::WRITE_TO_STDOUT>();
  port.send_n(data, size);
  port.recv([&](rpc::Buffer *buffer) {
    ret = reinterpret_cast<int *>(buffer->data)[0];
  });
  port.close();
  return ret;
}

int write_to_stderr(const void *data, size_t size) {
  int ret = 0;
  rpc::Client::Port port = rpc::client.open<rpc::WRITE_TO_STDERR>();
  port.send_n(data, size);
  port.recv([&](rpc::Buffer *buffer) {
    ret = reinterpret_cast<int *>(buffer->data)[0];
  });
  port.close();
  return ret;
}

int write_to_stream(uintptr_t file, const void *data, size_t size) {
  int ret = 0;
  rpc::Client::Port port = rpc::client.open<rpc::WRITE_TO_STREAM>();
  port.send([&](rpc::Buffer *buffer) {
    reinterpret_cast<uintptr_t *>(buffer->data)[0] = file;
  });
  port.send_n(data, size);
  port.recv([&](rpc::Buffer *buffer) {
    ret = reinterpret_cast<int *>(buffer->data)[0];
  });
  port.close();
  return ret;
}

FileIOResult write_func(File *f, const void *data, size_t size) {
  auto *gpu_file = reinterpret_cast<GPUFile *>(f);
  int ret = 0;
  if (gpu_file == stdout)
    ret = write_to_stdout(data, size);
  else if (gpu_file == stderr)
    ret = write_to_stderr(data, size);
  else
    ret = write_to_stream(gpu_file->get_file(), data, size);
  if (ret < 0)
    return {0, -ret};
  return ret;
}

} // namespace

static GPUFile StdIn(0UL, File::ModeFlags(File::OpenMode::READ));
File *stdin = &StdIn;

static GPUFile StdOut(0UL, File::ModeFlags(File::OpenMode::APPEND));
File *stdout = &StdOut;

static GPUFile StdErr(0UL, File::ModeFlags(File::OpenMode::APPEND));
File *stderr = &StdErr;

} // namespace __llvm_libc
