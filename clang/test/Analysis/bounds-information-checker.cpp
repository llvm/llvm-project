// RUN: %clang %s -std=c++20 -Xclang -verify --analyze \
// RUN:   -Xclang -analyzer-checker=core,alpha.cplusplus.BoundsInformation \
// RUN:   -Xclang -analyzer-checker=debug.ExprInspection

#include "std-span-system-header.h"

//----------------------------------------------------------------------------//
// std::span - no issue
//----------------------------------------------------------------------------//

void stdSpanFromCArrayNoIssue() {
  char buffer[4] = { '3', '.', '1', '4' };

  (void)std::span { buffer };                           // no-warning
  (void)std::span<char> { buffer, sizeof(buffer) };     // no-warning
  (void)std::span<char> { &buffer[0], sizeof(buffer) }; // no-warning

  char *ptr = buffer;
  size_t size = sizeof(buffer);
  (void)std::span<char> { ptr, size };                  // no-warning
}

void stdSpanFromStdArrayNoIssue() {
  std::array<char, 124> buffer { '3', '.', '1', '4' };

  (void)std::span { buffer }.first(4);                             // no-warning
  (void)std::span<char> { buffer.data(), buffer.size() }.first(4); // no-warning
  (void)std::span<char> { &buffer[0], buffer.size() }.first(4);    // no-warning

  char *ptr = buffer.data();
  size_t size = buffer.size();
  (void)std::span<char> { ptr, size }.first(4);                    // no-warning
}

void stdSpanFromStaticCArray() {
  static const uint8_t prefixDeltaFrame[6] = { 0x00, 0x00, 0x00, 0x01, 0x21, 0xe0 };
  (void)std::span<const uint8_t> { prefixDeltaFrame, sizeof(prefixDeltaFrame) };     // no-warning
  (void)std::span<const uint8_t> { &prefixDeltaFrame[0], sizeof(prefixDeltaFrame) }; // no-warning
}

// No issue because arguments from the same memory region.
// FIXME: This works locally, but not in actual WebKit code.
class MappedFileData {
public:
  unsigned size() const { return m_fileSize; }
  std::span<const uint8_t> span() const { return { static_cast<const uint8_t*>(m_fileData), size() }; } // no-warning
  std::span<uint8_t> mutableSpan() { return { static_cast<uint8_t*>(m_fileData), size() }; }            // no-warning

private:
  void* m_fileData { nullptr };
  unsigned m_fileSize { 0 };
};

//----------------------------------------------------------------------------//
// std::span - warnings
//----------------------------------------------------------------------------//

void stdSpanFromStdArrayWarnings() {
  std::array<char, 124> buffer { '3', '.', '1', '4' };
  (void)std::span<char> { &buffer[0], 4 };
  (void)std::span<char> { buffer.data(), 4 };
}

void stdSpanFromStdArrayOutOfBounds() {
  std::array<char, 4> buffer { '3', '.', '1', '4' };
  (void)std::span<char> { &buffer[0], 5 };    // expected-warning {{std::span constructed with overflow length}}
  (void)std::span<char> { buffer.data(), 5 }; // expected-warning {{std::span constructed with overflow length}}
}

struct HexNumberBuffer {
   std::array<char, 16> buffer;
   unsigned length;

   const char* characters() const { return &*(buffer.end() - length); }
   std::span<const char> span() const { return { characters(), length }; } // expected-warning {{std::span constructed from std::array with non-constant length}}
};
