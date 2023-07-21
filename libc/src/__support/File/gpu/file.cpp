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
#include "src/string/string_utils.h"

#include <stdio.h>

namespace __llvm_libc {

namespace {

FileIOResult write_func(File *, const void *, size_t);
int close_func(File *);

} // namespace

class GPUFile : public File {
  uintptr_t file;

public:
  constexpr GPUFile(uintptr_t file, File::ModeFlags modeflags)
      : File(&write_func, nullptr, nullptr, &close_func, nullptr, 0, _IONBF,
             false, modeflags),
        file(file) {}

  uintptr_t get_file() const { return file; }
};

namespace {

int write_to_stdout(const void *data, size_t size) {
  int ret = 0;
  rpc::Client::Port port = rpc::client.open<RPC_WRITE_TO_STDOUT>();
  port.send_n(data, size);
  port.recv([&](rpc::Buffer *buffer) {
    ret = reinterpret_cast<int *>(buffer->data)[0];
  });
  port.close();
  return ret;
}

int write_to_stderr(const void *data, size_t size) {
  int ret = 0;
  rpc::Client::Port port = rpc::client.open<RPC_WRITE_TO_STDERR>();
  port.send_n(data, size);
  port.recv([&](rpc::Buffer *buffer) {
    ret = reinterpret_cast<int *>(buffer->data)[0];
  });
  port.close();
  return ret;
}

int write_to_stream(uintptr_t file, const void *data, size_t size) {
  int ret = 0;
  rpc::Client::Port port = rpc::client.open<RPC_WRITE_TO_STREAM>();
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

int close_func(File *file) {
  int ret = 0;
  GPUFile *gpu_file = reinterpret_cast<GPUFile *>(file);
  rpc::Client::Port port = rpc::client.open<RPC_CLOSE_FILE>();
  port.send_and_recv(
      [=](rpc::Buffer *buffer) { buffer->data[0] = gpu_file->get_file(); },
      [&](rpc::Buffer *buffer) { ret = buffer->data[0]; });
  port.close();

  return ret;
}

} // namespace

void *ptr;

ErrorOr<File *> openfile(const char *path, const char *mode) {
  auto modeflags = File::mode_flags(mode);
  if (modeflags == 0)
    return Error(EINVAL);

  uintptr_t file;
  rpc::Client::Port port = rpc::client.open<RPC_OPEN_FILE>();
  port.send_n(path, internal::string_length(path) + 1);
  port.send_and_recv(
      [=](rpc::Buffer *buffer) {
        inline_memcpy(buffer->data, mode, internal::string_length(mode) + 1);
      },
      [&](rpc::Buffer *buffer) { file = buffer->data[0]; });
  port.close();

  static GPUFile gpu_file(0, 0);
  gpu_file = GPUFile(file, modeflags);
  return &gpu_file;
}

static GPUFile StdIn(0UL, File::ModeFlags(File::OpenMode::READ));
File *stdin = &StdIn;

static GPUFile StdOut(0UL, File::ModeFlags(File::OpenMode::APPEND));
File *stdout = &StdOut;

static GPUFile StdErr(0UL, File::ModeFlags(File::OpenMode::APPEND));
File *stderr = &StdErr;

} // namespace __llvm_libc

// Provide the external defintitions of the standard IO streams.
extern "C" {
FILE *stdin = reinterpret_cast<FILE *>(&__llvm_libc::StdIn);
FILE *stderr = reinterpret_cast<FILE *>(&__llvm_libc::StdErr);
FILE *stdout = reinterpret_cast<FILE *>(&__llvm_libc::StdOut);
}
