//===-- runtime/buffer.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// External file buffering

#ifndef FORTRAN_RUNTIME_BUFFER_H_
#define FORTRAN_RUNTIME_BUFFER_H_

#include "io-error.h"
#include "memory.h"
#include <algorithm>
#include <cinttypes>
#include <cstring>

namespace Fortran::runtime::io {

void LeftShiftBufferCircularly(char *, std::size_t bytes, std::size_t shift);

// Maintains a view of a contiguous region of a file in a memory buffer.
// The valid data in the buffer may be circular, but any active frame
// will also be contiguous in memory.  The requirement stems from the need to
// preserve read data that may be reused by means of Tn/TLn edit descriptors
// without needing to position the file (which may not always be possible,
// e.g. a socket) and a general desire to reduce system call counts.
template<typename STORE> class FileFrame {
public:
  using FileOffset = std::int64_t;

  ~FileFrame() { FreeMemoryAndNullify(buffer_); }

  // The valid data in the buffer begins at buffer_[start_] and proceeds
  // with possible wrap-around for length_ bytes.  The current frame
  // is offset by frame_ bytes into that region and is guaranteed to
  // be contiguous for at least as many bytes as were requested.

  FileOffset FrameAt() const { return fileOffset_ + frame_; }
  char *Frame() const { return buffer_ + start_ + frame_; }
  std::size_t FrameLength() const {
    return std::min<std::size_t>(length_ - frame_, size_ - (start_ + frame_));
  }
  std::size_t BytesBufferedBeforeFrame() const { return frame_ - start_; }

  // Returns a short frame at a non-fatal EOF.  Can return a long frame as well.
  std::size_t ReadFrame(
      FileOffset at, std::size_t bytes, IoErrorHandler &handler) {
    Flush(handler);
    Reallocate(bytes, handler);
    if (at < fileOffset_ || at > fileOffset_ + length_) {
      Reset(at);
    }
    frame_ = at - fileOffset_;
    if (static_cast<std::int64_t>(start_ + frame_ + bytes) > size_) {
      DiscardLeadingBytes(frame_, handler);
      if (static_cast<std::int64_t>(start_ + bytes) > size_) {
        // Frame would wrap around; shift current data (if any) to force
        // contiguity.
        RUNTIME_CHECK(handler, length_ < size_);
        if (start_ + length_ <= size_) {
          // [......abcde..] -> [abcde........]
          std::memmove(buffer_, buffer_ + start_, length_);
        } else {
          // [cde........ab] -> [abcde........]
          auto n{start_ + length_ - size_};  // 3 for cde
          RUNTIME_CHECK(handler, length_ >= n);
          std::memmove(buffer_ + n, buffer_ + start_, length_ - n);  // cdeab
          LeftShiftBufferCircularly(buffer_, length_, n);  // abcde
        }
        start_ = 0;
      }
    }
    while (FrameLength() < bytes) {
      auto next{start_ + length_};
      RUNTIME_CHECK(handler, next < size_);
      auto minBytes{bytes - FrameLength()};
      auto maxBytes{size_ - next};
      auto got{Store().Read(
          fileOffset_ + length_, buffer_ + next, minBytes, maxBytes, handler)};
      length_ += got;
      RUNTIME_CHECK(handler, length_ < size_);
      if (got < minBytes) {
        break;  // error or EOF & program can handle it
      }
    }
    return FrameLength();
  }

  void WriteFrame(FileOffset at, std::size_t bytes, IoErrorHandler &handler) {
    if (!dirty_ || at < fileOffset_ || at > fileOffset_ + length_ ||
        start_ + (at - fileOffset_) + static_cast<std::int64_t>(bytes) >
            size_) {
      Flush(handler);
      fileOffset_ = at;
      Reallocate(bytes, handler);
    }
    dirty_ = true;
    frame_ = at - fileOffset_;
    length_ = std::max<std::int64_t>(length_, frame_ + bytes);
  }

  void Flush(IoErrorHandler &handler) {
    if (dirty_) {
      while (length_ > 0) {
        std::size_t chunk{std::min<std::size_t>(length_, size_ - start_)};
        std::size_t put{
            Store().Write(fileOffset_, buffer_ + start_, chunk, handler)};
        length_ -= put;
        start_ += put;
        fileOffset_ += put;
        if (put < chunk) {
          break;
        }
      }
      Reset(fileOffset_);
    }
  }

private:
  STORE &Store() { return static_cast<STORE &>(*this); }

  void Reallocate(std::int64_t bytes, const Terminator &terminator) {
    if (bytes > size_) {
      char *old{buffer_};
      auto oldSize{size_};
      size_ = std::max<std::int64_t>(bytes, minBuffer);
      buffer_ =
          reinterpret_cast<char *>(AllocateMemoryOrCrash(terminator, size_));
      auto chunk{std::min<std::int64_t>(length_, oldSize - start_)};
      std::memcpy(buffer_, old + start_, chunk);
      start_ = 0;
      std::memcpy(buffer_ + chunk, old, length_ - chunk);
      FreeMemory(old);
    }
  }

  void Reset(FileOffset at) {
    start_ = length_ = frame_ = 0;
    fileOffset_ = at;
    dirty_ = false;
  }

  void DiscardLeadingBytes(std::int64_t n, const Terminator &terminator) {
    RUNTIME_CHECK(terminator, length_ >= n);
    length_ -= n;
    if (length_ == 0) {
      start_ = 0;
    } else {
      start_ += n;
      if (start_ >= size_) {
        start_ -= size_;
      }
    }
    if (frame_ >= n) {
      frame_ -= n;
    } else {
      frame_ = 0;
    }
    fileOffset_ += n;
  }

  static constexpr std::size_t minBuffer{64 << 10};

  char *buffer_{nullptr};
  std::int64_t size_{0};  // current allocated buffer size
  FileOffset fileOffset_{0};  // file offset corresponding to buffer valid data
  std::int64_t start_{0};  // buffer_[] offset of valid data
  std::int64_t length_{0};  // valid data length (can wrap)
  std::int64_t frame_{0};  // offset of current frame in valid data
  bool dirty_{false};
};
}
#endif  // FORTRAN_RUNTIME_BUFFER_H_
