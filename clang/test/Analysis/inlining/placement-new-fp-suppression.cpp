// RUN: %clang_analyze_cc1 -std=c++14 \
// RUN:  -analyzer-checker=core.CallAndMessage \
// RUN:  -analyzer-config suppress-null-return-paths=false \
// RUN:  -verify %s
// RUN: %clang_analyze_cc1 -std=c++14 \
// RUN:  -analyzer-checker=core.CallAndMessage \
// RUN:  -DSUPPRESSED \
// RUN:  -verify %s

#ifdef SUPPRESSED
// expected-no-diagnostics
#endif

#include <stdint.h>
#include "../Inputs/system-header-simulator-cxx.h"

void error();
void *malloc(size_t);


// From llvm/include/llvm/Support/MathExtras.h
inline uintptr_t alignAddr(const void *Addr, size_t Alignment) {
  return (((uintptr_t)Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1));
}

inline size_t alignmentAdjustment(const void *Ptr, size_t Alignment) {
  return alignAddr(Ptr, Alignment) - (uintptr_t)Ptr;
}


// From llvm/include/llvm/Support/MemAlloc.h
inline void *safe_malloc(size_t Sz) {
  void *Result = malloc(Sz);
  if (Result == nullptr)
    error();

  return Result;
}


// From llvm/include/llvm/Support/Allocator.h
class MallocAllocator {
public:
  void *Allocate(size_t Size, size_t /*Alignment*/) {
    return safe_malloc(Size);
  }
};

class BumpPtrAllocator {
public:
  void *Allocate(size_t Size, size_t Alignment) {
    BytesAllocated += Size;
    size_t Adjustment = alignmentAdjustment(CurPtr, Alignment);
    size_t SizeToAllocate = Size;

    size_t PaddedSize = SizeToAllocate + Alignment - 1;
    uintptr_t AlignedAddr = alignAddr(Allocator.Allocate(PaddedSize, 0),
                                      Alignment);
    char *AlignedPtr = (char*)AlignedAddr;

    return AlignedPtr;
  }

private:
  char *CurPtr = nullptr;
  size_t BytesAllocated = 0;
  MallocAllocator Allocator;
};


// From clang/include/clang/AST/ASTContextAllocate.h
class ASTContext;

void *operator new(size_t Bytes, const ASTContext &C, size_t Alignment = 8);
void *operator new[](size_t Bytes, const ASTContext &C, size_t Alignment = 8);


// From clang/include/clang/AST/ASTContext.h
class ASTContext {
public:
  void *Allocate(size_t Size, unsigned Align = 8) const {
    return BumpAlloc.Allocate(Size, Align);
  }

  template <typename T>
  T *Allocate(size_t Num = 1) const {
    return static_cast<T *>(Allocate(Num * sizeof(T), alignof(T)));
  }

private:
  mutable BumpPtrAllocator BumpAlloc;
};


// From clang/include/clang/AST/ASTContext.h
inline void *operator new(size_t Bytes, const ASTContext &C,
                          size_t Alignment /* = 8 */) {
  return C.Allocate(Bytes, Alignment);
}

inline void *operator new[](size_t Bytes, const ASTContext &C,
                            size_t Alignment /* = 8 */) {
  return C.Allocate(Bytes, Alignment);
}


// From clang/include/clang/AST/Attr.h
void *operator new(size_t Bytes, ASTContext &C,
                   size_t Alignment = 8) noexcept {
  return ::operator new(Bytes, C, Alignment);
}


class A {
public:
  void setValue(int value) { Value = value; }
private:
  int Value;
};

void f(const ASTContext &C) {
  A *a = new (C) A;
  a->setValue(13);
#ifndef SUPPRESSED
  // expected-warning@-2 {{Called C++ object pointer is null}}
#endif
}	

void g(const ASTContext &C) {
  A *a = new (C) A[1];
  a[0].setValue(13);
#ifndef SUPPRESSED
  // expected-warning@-2 {{Called C++ object pointer is null}}
#endif
}

