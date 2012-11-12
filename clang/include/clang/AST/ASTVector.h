//===- ASTVector.h - Vector that uses ASTContext for allocation  --*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides ASTVector, a vector  ADT whose contents are
//  allocated using the allocator associated with an ASTContext..
//
//===----------------------------------------------------------------------===//

// FIXME: Most of this is copy-and-paste from BumpVector.h and SmallVector.h.
// We can refactor this core logic into something common.

#ifndef LLVM_CLANG_AST_VECTOR
#define LLVM_CLANG_AST_VECTOR

#include "llvm/Support/type_traits.h"
#include "llvm/Support/Allocator.h"
#include "llvm/ADT/PointerIntPair.h"
#include <algorithm>
#include <memory>
#include <cstring>

#ifdef _MSC_VER
namespace std {
#if _MSC_VER <= 1310
  // Work around flawed VC++ implementation of std::uninitialized_copy.  Define
  // additional overloads so that elements with pointer types are recognized as
  // scalars and not objects, causing bizarre type conversion errors.
  template<class T1, class T2>
  inline _Scalar_ptr_iterator_tag _Ptr_cat(T1 **, T2 **) {
    _Scalar_ptr_iterator_tag _Cat;
    return _Cat;
  }

  template<class T1, class T2>
  inline _Scalar_ptr_iterator_tag _Ptr_cat(T1* const *, T2 **) {
    _Scalar_ptr_iterator_tag _Cat;
    return _Cat;
  }
#else
  // FIXME: It is not clear if the problem is fixed in VS 2005.  What is clear
  // is that the above hack won't work if it wasn't fixed.
#endif
}
#endif

namespace clang {

template<typename T>
class ASTVector {
  T *Begin, *End, *Capacity;

  void setEnd(T *P) { this->End = P; }

public:
  // Default ctor - Initialize to empty.
  explicit ASTVector(ASTContext &C, unsigned N = 0)
  : Begin(NULL), End(NULL), Capacity(NULL) {
    reserve(C, N);
  }

  ~ASTVector() {
    if (llvm::is_class<T>::value) {
      // Destroy the constructed elements in the vector.
      destroy_range(Begin, End);
    }
  }

  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T value_type;
  typedef T* iterator;
  typedef const T* const_iterator;

  typedef std::reverse_iterator<const_iterator>  const_reverse_iterator;
  typedef std::reverse_iterator<iterator>  reverse_iterator;

  typedef T& reference;
  typedef const T& const_reference;
  typedef T* pointer;
  typedef const T* const_pointer;

  // forward iterator creation methods.
  iterator begin() { return Begin; }
  const_iterator begin() const { return Begin; }
  iterator end() { return End; }
  const_iterator end() const { return End; }

  // reverse iterator creation methods.
  reverse_iterator rbegin()            { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const{ return const_reverse_iterator(end()); }
  reverse_iterator rend()              { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin());}

  bool empty() const { return Begin == End; }
  size_type size() const { return End-Begin; }

  reference operator[](unsigned idx) {
    assert(Begin + idx < End);
    return Begin[idx];
  }
  const_reference operator[](unsigned idx) const {
    assert(Begin + idx < End);
    return Begin[idx];
  }

  reference front() {
    return begin()[0];
  }
  const_reference front() const {
    return begin()[0];
  }

  reference back() {
    return end()[-1];
  }
  const_reference back() const {
    return end()[-1];
  }

  void pop_back() {
    --End;
    End->~T();
  }

  T pop_back_val() {
    T Result = back();
    pop_back();
    return Result;
  }

  void clear() {
    if (llvm::is_class<T>::value) {
      destroy_range(Begin, End);
    }
    End = Begin;
  }

  /// data - Return a pointer to the vector's buffer, even if empty().
  pointer data() {
    return pointer(Begin);
  }

  /// data - Return a pointer to the vector's buffer, even if empty().
  const_pointer data() const {
    return const_pointer(Begin);
  }

  void push_back(const_reference Elt, ASTContext &C) {
    if (End < Capacity) {
    Retry:
      new (End) T(Elt);
      ++End;
      return;
    }
    grow(C);
    goto Retry;
  }

  void reserve(ASTContext &C, unsigned N) {
    if (unsigned(Capacity-Begin) < N)
      grow(C, N);
  }

  /// capacity - Return the total number of elements in the currently allocated
  /// buffer.
  size_t capacity() const { return Capacity - Begin; }

  /// append - Add the specified range to the end of the SmallVector.
  ///
  template<typename in_iter>
  void append(ASTContext &C, in_iter in_start, in_iter in_end) {
    size_type NumInputs = std::distance(in_start, in_end);

    if (NumInputs == 0)
      return;

    // Grow allocated space if needed.
    if (NumInputs > size_type(this->capacity_ptr()-this->end()))
      this->grow(C, this->size()+NumInputs);

    // Copy the new elements over.
    // TODO: NEED To compile time dispatch on whether in_iter is a random access
    // iterator to use the fast uninitialized_copy.
    std::uninitialized_copy(in_start, in_end, this->end());
    this->setEnd(this->end() + NumInputs);
  }

  /// append - Add the specified range to the end of the SmallVector.
  ///
  void append(ASTContext &C, size_type NumInputs, const T &Elt) {
    // Grow allocated space if needed.
    if (NumInputs > size_type(this->capacity_ptr()-this->end()))
      this->grow(C, this->size()+NumInputs);

    // Copy the new elements over.
    std::uninitialized_fill_n(this->end(), NumInputs, Elt);
    this->setEnd(this->end() + NumInputs);
  }

  /// uninitialized_copy - Copy the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template<typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(I, E, Dest);
  }

  iterator insert(ASTContext &C, iterator I, const T &Elt) {
    if (I == this->end()) {  // Important special case for empty vector.
      push_back(Elt);
      return this->end()-1;
    }

    if (this->EndX < this->CapacityX) {
    Retry:
      new (this->end()) T(this->back());
      this->setEnd(this->end()+1);
      // Push everything else over.
      std::copy_backward(I, this->end()-1, this->end());
      *I = Elt;
      return I;
    }
    size_t EltNo = I-this->begin();
    this->grow(C);
    I = this->begin()+EltNo;
    goto Retry;
  }

  iterator insert(ASTContext &C, iterator I, size_type NumToInsert,
                  const T &Elt) {
    if (I == this->end()) {  // Important special case for empty vector.
      append(C, NumToInsert, Elt);
      return this->end()-1;
    }

    // Convert iterator to elt# to avoid invalidating iterator when we reserve()
    size_t InsertElt = I - this->begin();

    // Ensure there is enough space.
    reserve(C, static_cast<unsigned>(this->size() + NumToInsert));

    // Uninvalidate the iterator.
    I = this->begin()+InsertElt;

    // If there are more elements between the insertion point and the end of the
    // range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end()-I) >= NumToInsert) {
      T *OldEnd = this->end();
      append(C, this->end()-NumToInsert, this->end());

      // Copy the existing elements that get replaced.
      std::copy_backward(I, OldEnd-NumToInsert, OldEnd);

      std::fill_n(I, NumToInsert, Elt);
      return I;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Copy over the elements that we're about to overwrite.
    T *OldEnd = this->end();
    this->setEnd(this->end() + NumToInsert);
    size_t NumOverwritten = OldEnd-I;
    this->uninitialized_copy(I, OldEnd, this->end()-NumOverwritten);

    // Replace the overwritten part.
    std::fill_n(I, NumOverwritten, Elt);

    // Insert the non-overwritten middle part.
    std::uninitialized_fill_n(OldEnd, NumToInsert-NumOverwritten, Elt);
    return I;
  }

  template<typename ItTy>
  iterator insert(ASTContext &C, iterator I, ItTy From, ItTy To) {
    if (I == this->end()) {  // Important special case for empty vector.
      append(C, From, To);
      return this->end()-1;
    }

    size_t NumToInsert = std::distance(From, To);
    // Convert iterator to elt# to avoid invalidating iterator when we reserve()
    size_t InsertElt = I - this->begin();

    // Ensure there is enough space.
    reserve(C, static_cast<unsigned>(this->size() + NumToInsert));

    // Uninvalidate the iterator.
    I = this->begin()+InsertElt;

    // If there are more elements between the insertion point and the end of the
    // range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end()-I) >= NumToInsert) {
      T *OldEnd = this->end();
      append(C, this->end()-NumToInsert, this->end());

      // Copy the existing elements that get replaced.
      std::copy_backward(I, OldEnd-NumToInsert, OldEnd);

      std::copy(From, To, I);
      return I;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Copy over the elements that we're about to overwrite.
    T *OldEnd = this->end();
    this->setEnd(this->end() + NumToInsert);
    size_t NumOverwritten = OldEnd-I;
    this->uninitialized_copy(I, OldEnd, this->end()-NumOverwritten);

    // Replace the overwritten part.
    for (; NumOverwritten > 0; --NumOverwritten) {
      *I = *From;
      ++I; ++From;
    }

    // Insert the non-overwritten middle part.
    this->uninitialized_copy(From, To, OldEnd);
    return I;
  }

  void resize(ASTContext &C, unsigned N, const T &NV) {
    if (N < this->size()) {
      this->destroy_range(this->begin()+N, this->end());
      this->setEnd(this->begin()+N);
    } else if (N > this->size()) {
      if (this->capacity() < N)
        this->grow(C, N);
      construct_range(this->end(), this->begin()+N, NV);
      this->setEnd(this->begin()+N);
    }
  }

private:
  /// grow - double the size of the allocated memory, guaranteeing space for at
  /// least one more element or MinSize if specified.
  void grow(ASTContext &C, size_type MinSize = 1);

  void construct_range(T *S, T *E, const T &Elt) {
    for (; S != E; ++S)
      new (S) T(Elt);
  }

  void destroy_range(T *S, T *E) {
    while (S != E) {
      --E;
      E->~T();
    }
  }

protected:
  iterator capacity_ptr() { return (iterator)this->Capacity; }
};

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T>
void ASTVector<T>::grow(ASTContext &C, size_t MinSize) {
  size_t CurCapacity = Capacity-Begin;
  size_t CurSize = size();
  size_t NewCapacity = 2*CurCapacity;
  if (NewCapacity < MinSize)
    NewCapacity = MinSize;

  // Allocate the memory from the ASTContext.
  T *NewElts = new (C, llvm::alignOf<T>()) T[NewCapacity];

  // Copy the elements over.
  if (llvm::is_class<T>::value) {
    std::uninitialized_copy(Begin, End, NewElts);
    // Destroy the original elements.
    destroy_range(Begin, End);
  }
  else {
    // Use memcpy for PODs (std::uninitialized_copy optimizes to memmove).
    memcpy(NewElts, Begin, CurSize * sizeof(T));
  }

  // ASTContext never frees any memory.
  Begin = NewElts;
  End = NewElts+CurSize;
  Capacity = Begin+NewCapacity;
}

} // end: clang namespace
#endif
