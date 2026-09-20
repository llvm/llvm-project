//===--- InterpStack.h - Stack implementation for the VM --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the upwards-growing stack used by the interpreter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_INTERPSTACK_H
#define LLVM_CLANG_AST_INTERP_INTERPSTACK_H

#include "FixedPoint.h"
#include "IntegralAP.h"
#include "MemberPointer.h"
#include "PrimType.h"

namespace clang {
namespace interp {

template <class T>
struct datasizeof_impl {
  LLVM_NO_UNIQUE_ADDRESS T v;
  char first_padding_byte;
};

template <class T>
constexpr size_t datasizeof_v = offsetof(datasizeof_impl<T>, first_padding_byte);

/// Stack frame storing temporaries and parameters.
class InterpStack final {
public:
  InterpStack() = default;

  /// Destroys the stack, freeing up storage.
  ~InterpStack();

  template <size_t N> struct Padding {
    char padding[N];
  };

  template <> struct Padding<0> {};

  template <class T> struct alignas(void *) StackFrame {
    static_assert(alignof(T) <= alignof(void *),
                  "Unexpected overaligned object");

    template <class... Args>
    StackFrame(Args &&...args)
        : v(std::forward<Args>(args)...), type(toPrimType<T>()) {}

    static constexpr size_t getPaddingSize() {
      if constexpr (sizeof(T) < sizeof(void*))
        return sizeof(void*) - datasizeof_v<T> - 1;
      else if constexpr (sizeof(T) == datasizeof_v<T>)
        return sizeof(void*) - 1;
      else
        return sizeof(T) - datasizeof_v<T> - 1;
    }

    LLVM_NO_UNIQUE_ADDRESS T v;
    LLVM_NO_UNIQUE_ADDRESS Padding<getPaddingSize()> padding;
    PrimType type;
  };

  /// Constructs a value in place on the top of the stack.
  template <typename T, typename... Tys> void push(Tys &&...Args) {
    using Frame = StackFrame<T>;
    new (grow<sizeof(Frame)>()) Frame(std::forward<Tys>(Args)...);
  }

  /// Returns the value from the top of the stack and removes it.
  template <typename T> T pop() {
    assert(getNextObjectType() == toPrimType<T>());
    T *Ptr = &peekInternal<T>();
    T Value = std::move(*Ptr);
    shrink(sizeof(StackFrame<T>));
    return Value;
  }

  /// Discards the top value from the stack.
  template <typename T> void discard() {
    assert(getNextObjectType() == toPrimType<T>());
    T *Ptr = &peekInternal<T>();
    if constexpr (!std::is_trivially_destructible_v<T>) {
      Ptr->~T();
    }
    shrink(sizeof(StackFrame<T>));
  }
  void discardSlow();

  /// Returns a reference to the value on the top of the stack.
  template <typename T> T &peek() const {
    assert(getNextObjectType() == toPrimType<T>());
    return peekInternal<T>();
  }

  template <typename T> T &peek(size_t Offset) const {
    assert(aligned(Offset));
    return *reinterpret_cast<T *>(peekData(Offset));
  }

  /// Returns a pointer to the top object.
  void *top() const { return Chunk ? peekData(0) : nullptr; }

  /// Returns the size of the stack in bytes.
  size_t size() const { return StackSize; }

  /// Clears the stack.
  void clear();
  void clearTo(size_t NewSize);

  /// Returns whether the stack is empty.
  bool empty() const { return StackSize == 0; }

  /// dump the stack contents to stderr.
  void dump() const;

private:
  PrimType getNextObjectType() const {
    return *static_cast<PrimType *>(peekData(1));
  }

  /// Like the public peek(), but without the debug type checks.
  template <typename T> T &peekInternal() const {
    return static_cast<StackFrame<T> *>(peekData(sizeof(StackFrame<T>)))->v;
  }

  /// Grows the stack to accommodate a value and returns a pointer to it.
  template <size_t Size> void *grow() {
    assert(Size < ChunkSize - sizeof(StackChunk) && "Object too large");
    static_assert(aligned(Size));

    // Allocate a new stack chunk if necessary.
    if (LLVM_UNLIKELY(!Chunk)) {
      Chunk = new (std::malloc(ChunkSize)) StackChunk(Chunk);
    } else if (LLVM_UNLIKELY(Chunk->size() >
                             ChunkSize - sizeof(StackChunk) - Size)) {
      if (Chunk->Next) {
        Chunk = Chunk->Next;
      } else {
        StackChunk *Next = new (std::malloc(ChunkSize)) StackChunk(Chunk);
        Chunk->Next = Next;
        Chunk = Next;
      }
    }

    auto *Object = reinterpret_cast<void *>(Chunk->start() + Chunk->Size);
    Chunk->Size += Size;
    StackSize += Size;
    return Object;
  }

  /// Returns a pointer from the top of the stack.
  void *peekData(size_t Size) const;
  /// Shrinks the stack.
  void shrink(size_t Size);

  /// Allocate stack space in 1Mb chunks.
  static constexpr size_t ChunkSize = 1024 * 1024;

  /// Metadata for each stack chunk.
  ///
  /// The stack is composed of a linked list of chunks. Whenever an allocation
  /// is out of bounds, a new chunk is linked. When a chunk becomes empty,
  /// it is not immediately freed: a chunk is deallocated only when the
  /// predecessor becomes empty.
  struct StackChunk {
    StackChunk *Next;
    StackChunk *Prev;
    uint32_t Size;

    StackChunk(StackChunk *Prev = nullptr)
        : Next(nullptr), Prev(Prev), Size(0) {}

    /// Returns the size of the chunk, minus the header.
    size_t size() const { return Size; }

    /// Returns a pointer to the start of the data region.
    char *start() { return reinterpret_cast<char *>(this + 1); }
    const char *start() const {
      return reinterpret_cast<const char *>(this + 1);
    }
  };
  static_assert(sizeof(StackChunk) < ChunkSize, "Invalid chunk size");

  /// First chunk on the stack.
  StackChunk *Chunk = nullptr;
  /// Total size of the stack.
  size_t StackSize = 0;

  template <typename T> static constexpr PrimType toPrimType() {
    if constexpr (std::is_same_v<T, Pointer>)
      return PT_Ptr;
    else if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, Boolean>)
      return PT_Bool;
    else if constexpr (std::is_same_v<T, int8_t> ||
                       std::is_same_v<T, Char<true>>)
      return PT_Sint8;
    else if constexpr (std::is_same_v<T, uint8_t> ||
                       std::is_same_v<T, Char<false>>)
      return PT_Uint8;
    else if constexpr (std::is_same_v<T, Integral<16, true>>)
      return PT_Sint16;
    else if constexpr (std::is_same_v<T, Integral<16, false>>)
      return PT_Uint16;
    else if constexpr (std::is_same_v<T, Integral<32, true>>)
      return PT_Sint32;
    else if constexpr (std::is_same_v<T, Integral<32, false>>)
      return PT_Uint32;
    else if constexpr (std::is_same_v<T, Integral<64, true>>)
      return PT_Sint64;
    else if constexpr (std::is_same_v<T, Integral<64, false>>)
      return PT_Uint64;

    else if constexpr (std::is_same_v<T, Floating>)
      return PT_Float;
    else if constexpr (std::is_same_v<T, IntegralAP<true>>)
      return PT_IntAP;
    else if constexpr (std::is_same_v<T, IntegralAP<false>>)
      return PT_IntAP;
    else if constexpr (std::is_same_v<T, MemberPointer>)
      return PT_MemberPtr;
    else if constexpr (std::is_same_v<T, FixedPoint>)
      return PT_FixedPoint;

    llvm_unreachable("unknown type push()'ed into InterpStack");
  }
};

} // namespace interp
} // namespace clang

#endif
