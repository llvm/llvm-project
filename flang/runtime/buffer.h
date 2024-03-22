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

#include "freestanding-tools.h"
#include "io-error.h"
#include "flang/Runtime/memory.h"
#include <algorithm>
#include <cinttypes>
#include <cstring>

namespace Fortran::runtime::io {

RT_API_ATTRS void LeftShiftBufferCircularly(
    char *, std::size_t bytes, std::size_t shift);

// Maintains a view of a contiguous region of a file in a memory buffer.
// The valid data in the buffer may be circular, but any active frame
// will also be contiguous in memory.  The requirement stems from the need to
// preserve read data that may be reused by means of Tn/TLn edit descriptors
// without needing to position the file (which may not always be possible,
// e.g. a socket) and a general desire to reduce system call counts.
//
// Possible scenario with a tiny 32-byte buffer after a ReadFrame or
// WriteFrame with a file offset of 103 to access "DEF":
//
//    fileOffset_ 100 --+  +-+ frame of interest (103:105)
//   file:  ............ABCDEFGHIJKLMNOPQRSTUVWXYZ....
// buffer: [NOPQRSTUVWXYZ......ABCDEFGHIJKLM]   (size_ == 32)
//                             |  +-- frame_ == 3
//                             +----- start_ == 19, length_ == 26
//
// The buffer holds length_ == 26 bytes from file offsets 100:125.
// Those 26 bytes "wrap around" the end of the circular buffer,
// so file offsets 100:112 map to buffer offsets 19:31 ("A..M") and
//    file offsets 113:125 map to buffer offsets  0:12 ("N..Z")
// The 3-byte frame of file offsets 103:105 is contiguous in the buffer
// at buffer offset (start_ + frame_) == 22 ("DEF").

template <typename STORE, std::size_t minBuffer = 65536> class FileFrame {
public:
  using FileOffset = std::int64_t;

  RT_API_ATTRS ~FileFrame() { FreeMemoryAndNullify(buffer_); }

  // The valid data in the buffer begins at buffer_[start_] and proceeds
  // with possible wrap-around for length_ bytes.  The current frame
  // is offset by frame_ bytes into that region and is guaranteed to
  // be contiguous for at least as many bytes as were requested.

  RT_API_ATTRS FileOffset FrameAt() const { return fileOffset_ + frame_; }
  RT_API_ATTRS char *Frame() const { return buffer_ + start_ + frame_; }
  RT_API_ATTRS std::size_t FrameLength() const {
    return std::min<std::size_t>(length_ - frame_, size_ - (start_ + frame_));
  }
  RT_API_ATTRS std::size_t BytesBufferedBeforeFrame() const {
    return frame_ - start_;
  }

  // Returns a short frame at a non-fatal EOF.  Can return a long frame as well.
  RT_API_ATTRS std::size_t ReadFrame(
      FileOffset at, std::size_t bytes, IoErrorHandler &handler) {
    Flush(handler);
    Reallocate(bytes, handler);
    std::int64_t newFrame{at - fileOffset_};
    if (newFrame < 0 || newFrame > length_) {
      Reset(at);
    } else {
      frame_ = newFrame;
    }
    RUNTIME_CHECK(handler, at == fileOffset_ + frame_);
    if (static_cast<std::int64_t>(start_ + frame_ + bytes) > size_) {
      DiscardLeadingBytes(frame_, handler);
      MakeDataContiguous(handler, bytes);
      RUNTIME_CHECK(handler, at == fileOffset_ + frame_);
    }
    if (FrameLength() < bytes) {
      auto next{start_ + length_};
      RUNTIME_CHECK(handler, next < size_);
      auto minBytes{bytes - FrameLength()};
      auto maxBytes{size_ - next};
      auto got{Store().Read(
          fileOffset_ + length_, buffer_ + next, minBytes, maxBytes, handler)};
      length_ += got;
      RUNTIME_CHECK(handler, length_ <= size_);
    }
    return FrameLength();
  }

  RT_API_ATTRS void WriteFrame(
      FileOffset at, std::size_t bytes, IoErrorHandler &handler) {
    Reallocate(bytes, handler);
    std::int64_t newFrame{at - fileOffset_};
    if (!dirty_ || newFrame < 0 || newFrame > length_) {
      Flush(handler);
      Reset(at);
    } else if (start_ + newFrame + static_cast<std::int64_t>(bytes) > size_) {
      // Flush leading data before "at", retain from "at" onward
      Flush(handler, length_ - newFrame);
      MakeDataContiguous(handler, bytes);
    } else {
      frame_ = newFrame;
    }
    RUNTIME_CHECK(handler, at == fileOffset_ + frame_);
    dirty_ = true;
    length_ = std::max<std::int64_t>(length_, frame_ + bytes);
  }

  RT_API_ATTRS void Flush(IoErrorHandler &handler, std::int64_t keep = 0) {
    if (dirty_) {
      while (length_ > keep) {
        std::size_t chunk{
            std::min<std::size_t>(length_ - keep, size_ - start_)};
        std::size_t put{
            Store().Write(fileOffset_, buffer_ + start_, chunk, handler)};
        DiscardLeadingBytes(put, handler);
        if (put < chunk) {
          break;
        }
      }
      if (length_ == 0) {
        Reset(fileOffset_);
      }
    }
  }

  RT_API_ATTRS void TruncateFrame(std::int64_t at, IoErrorHandler &handler) {
    RUNTIME_CHECK(handler, !dirty_);
    if (at <= fileOffset_) {
      Reset(at);
    } else if (at < fileOffset_ + length_) {
      length_ = at - fileOffset_;
    }
  }

private:
  RT_API_ATTRS STORE &Store() { return static_cast<STORE &>(*this); }

  RT_API_ATTRS void Reallocate(
      std::int64_t bytes, const Terminator &terminator) {
    if (bytes > size_) {
      char *old{buffer_};
      auto oldSize{size_};
      size_ = std::max<std::int64_t>(bytes, size_ + minBuffer);
      buffer_ =
          reinterpret_cast<char *>(AllocateMemoryOrCrash(terminator, size_));
      auto chunk{std::min<std::int64_t>(length_, oldSize - start_)};
      // "memcpy" in glibc has a "nonnull" attribute on the source pointer.
      // Avoid passing a null pointer, since it would result in an undefined
      // behavior.
      if (old != nullptr) {
        std::memcpy(buffer_, old + start_, chunk);
        std::memcpy(buffer_ + chunk, old, length_ - chunk);
        FreeMemory(old);
      }
      start_ = 0;
    }
  }

  RT_API_ATTRS void Reset(FileOffset at) {
    start_ = length_ = frame_ = 0;
    fileOffset_ = at;
    dirty_ = false;
  }

  RT_API_ATTRS void DiscardLeadingBytes(
      std::int64_t n, const Terminator &terminator) {
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

  RT_API_ATTRS void MakeDataContiguous(
      IoErrorHandler &handler, std::size_t bytes) {
    if (static_cast<std::int64_t>(start_ + bytes) > size_) {
      // Frame would wrap around; shift current data (if any) to force
      // contiguity.
      RUNTIME_CHECK(handler, length_ < size_);
      if (start_ + length_ <= size_) {
        // [......abcde..] -> [abcde........]
        runtime::memmove(buffer_, buffer_ + start_, length_);
      } else {
        // [cde........ab] -> [abcde........]
        auto n{start_ + length_ - size_}; // 3 for cde
        RUNTIME_CHECK(handler, length_ >= n);
        runtime::memmove(buffer_ + n, buffer_ + start_, length_ - n); // cdeab
        LeftShiftBufferCircularly(buffer_, length_, n); // abcde
      }
      start_ = 0;
    }
  }

  char *buffer_{nullptr};
  std::int64_t size_{0}; // current allocated buffer size
  FileOffset fileOffset_{0}; // file offset corresponding to buffer valid data
  std::int64_t start_{0}; // buffer_[] offset of valid data
  std::int64_t length_{0}; // valid data length (can wrap)
  std::int64_t frame_{0}; // offset of current frame in valid data
  bool dirty_{false};
};
} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_BUFFER_H_
