#include "../../runtime/buffer.h"
#include "testing.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>

static constexpr std::size_t tinyBuffer{32};
using FileOffset = std::int64_t;
using namespace Fortran::runtime;
using namespace Fortran::runtime::io;

class Store : public FileFrame<Store, tinyBuffer> {
public:
  explicit Store(std::size_t bytes = 65536) : bytes_{bytes} {
    data_.reset(new char[bytes]);
    std::memset(&data_[0], 0, bytes);
  }
  std::size_t bytes() const { return bytes_; }
  void set_enforceSequence(bool yes = true) { enforceSequence_ = yes; }
  void set_expect(FileOffset to) { expect_ = to; }

  std::size_t Read(FileOffset at, char *to, std::size_t minBytes,
      std::size_t maxBytes, IoErrorHandler &handler) {
    if (enforceSequence_ && at != expect_) {
      handler.SignalError("Read(%d,%d,%d) not at expected %d",
          static_cast<int>(at), static_cast<int>(minBytes),
          static_cast<int>(maxBytes), static_cast<int>(expect_));
    } else if (at < 0 || at + minBytes > bytes_) {
      handler.SignalError("Read(%d,%d,%d) is out of bounds",
          static_cast<int>(at), static_cast<int>(minBytes),
          static_cast<int>(maxBytes));
    }
    auto result{std::min(maxBytes, bytes_ - at)};
    std::memcpy(to, &data_[at], result);
    expect_ = at + result;
    return result;
  }
  std::size_t Write(FileOffset at, const char *from, std::size_t bytes,
      IoErrorHandler &handler) {
    if (enforceSequence_ && at != expect_) {
      handler.SignalError("Write(%d,%d) not at expected %d",
          static_cast<int>(at), static_cast<int>(bytes),
          static_cast<int>(expect_));
    } else if (at < 0 || at + bytes > bytes_) {
      handler.SignalError("Write(%d,%d) is out of bounds", static_cast<int>(at),
          static_cast<int>(bytes));
    }
    std::memcpy(&data_[at], from, bytes);
    expect_ = at + bytes;
    return bytes;
  }

private:
  std::size_t bytes_;
  std::unique_ptr<char[]> data_;
  bool enforceSequence_{false};
  FileOffset expect_{0};
};

inline int ChunkSize(int j, int most) {
  // 31, 1, 29, 3, 27, ...
  j %= tinyBuffer;
  auto chunk{
      static_cast<int>(((j % 2) ? j : (tinyBuffer - 1 - j)) % tinyBuffer)};
  return std::min(chunk, most);
}

inline int ValueFor(int at) { return (at ^ (at >> 8)) & 0xff; }

int main() {
  StartTests();
  Terminator terminator{__FILE__, __LINE__};
  IoErrorHandler handler{terminator};
  Store store;
  store.set_enforceSequence(true);
  const auto bytes{static_cast<FileOffset>(store.bytes())};
  // Fill with an assortment of chunks
  int at{0}, j{0};
  while (at < bytes) {
    auto chunk{ChunkSize(j, static_cast<int>(bytes - at))};
    store.WriteFrame(at, chunk, handler);
    char *to{store.Frame()};
    for (int k{0}; k < chunk; ++k) {
      to[k] = ValueFor(at + k);
    }
    at += chunk;
    ++j;
  }
  store.Flush(handler);
  // Validate
  store.set_expect(0);
  at = 0;
  while (at < bytes) {
    auto chunk{ChunkSize(j, static_cast<int>(bytes - at))};
    std::size_t frame{store.ReadFrame(at, chunk, handler)};
    if (frame < static_cast<std::size_t>(chunk)) {
      Fail() << "Badly-sized ReadFrame at " << at << ", chunk=" << chunk
             << ", got " << frame << '\n';
      break;
    }
    const char *from{store.Frame()};
    for (int k{0}; k < chunk; ++k) {
      auto expect{static_cast<char>(ValueFor(at + k))};
      if (from[k] != expect) {
        Fail() << "At " << at << '+' << k << '(' << (at + k) << "), read "
               << (from[k] & 0xff) << ", expected " << static_cast<int>(expect)
               << '\n';
      }
    }
    at += chunk;
    ++j;
  }
  return EndTests();
}
