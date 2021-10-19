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

#include <memory>

namespace clang {
namespace interp {

/// Stack frame storing temporaries and parameters.
class InterpStack final {
public:
  InterpStack() {}

  /// Destroys the stack, freeing up storage.
  ~InterpStack();

  /// Constructs a value in place on the top of the stack.
  template <typename T, typename... Tys> void push(Tys &&... Args) {
    new (grow(aligned_size<T>())) T(std::forward<Tys>(Args)...);
  }

  /// Returns the value from the top of the stack and removes it.
  template <typename T> T pop() {
    auto *Ptr = &peek<T>();
    auto Value = std::move(*Ptr);
    Ptr->~T();
    shrink(aligned_size<T>());
    return Value;
  }

  /// Discards the top value from the stack.
  template <typename T> void discard() {
    auto *Ptr = &peek<T>();
    Ptr->~T();
    shrink(aligned_size<T>());
  }

  /// Returns a reference to the value on the top of the stack.
  template <typename T> T &peek() {
    return *reinterpret_cast<T *>(peek(aligned_size<T>()));
  }

  /// Returns a pointer to the top object.
  void *top() { return Chunk ? peek(0) : nullptr; }

  /// Returns the size of the stack in bytes.
  size_t size() const { return StackSize; }

  /// Clears the stack without calling any destructors.
  void clear();

private:
  /// All stack slots are aligned to the native pointer alignment for storage.
  /// The size of an object is rounded up to a pointer alignment multiple.
  template <typename T> constexpr size_t aligned_size() const {
    constexpr size_t PtrAlign = alignof(void *);
    return ((sizeof(T) + PtrAlign - 1) / PtrAlign) * PtrAlign;
  }

  /// Grows the stack to accommodate a value and returns a pointer to it.
  void *grow(size_t Size);
  /// Returns a pointer from the top of the stack.
  void *peek(size_t Size);
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
    char *End;

    StackChunk(StackChunk *Prev = nullptr)
        : Next(nullptr), Prev(Prev), End(reinterpret_cast<char *>(this + 1)) {}

    /// Returns the size of the chunk, minus the header.
    size_t size() { return End - start(); }

    /// Returns a pointer to the start of the data region.
    char *start() { return reinterpret_cast<char *>(this + 1); }
  };
  static_assert(sizeof(StackChunk) < ChunkSize, "Invalid chunk size");

  /// First chunk on the stack.
  StackChunk *Chunk = nullptr;
  /// Total size of the stack.
  size_t StackSize = 0;
};

} // namespace interp
} // namespace clang

#endif
