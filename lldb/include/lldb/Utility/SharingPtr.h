//===---------------------SharingPtr.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef utility_SharingPtr_h_
#define utility_SharingPtr_h_

#include <memory>

// Microsoft Visual C++ currently does not enable std::atomic to work in CLR
// mode - as such we need to "hack around it" for MSVC++ builds only using
// Windows specific intrinsics instead of the C++11 atomic support
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <atomic>
#endif

#include <stddef.h>


//#define ENABLE_SP_LOGGING 1 // DON'T CHECK THIS LINE IN UNLESS COMMENTED OUT
#if defined(ENABLE_SP_LOGGING)

extern "C" void track_sp(void *sp_this, void *ptr, long count);

#endif

namespace lldb_private {

namespace imp {

class shared_count {
  shared_count(const shared_count &) = delete;
  shared_count &operator=(const shared_count &) = delete;

public:
  explicit shared_count(long refs = 0) : shared_owners_(refs) {}

  void add_shared();
  void release_shared();
  long use_count() const { return shared_owners_ + 1; }

protected:
#ifdef _MSC_VER
  long shared_owners_;
#else
  std::atomic<long> shared_owners_;
#endif
  virtual ~shared_count();

private:
  virtual void on_zero_shared() = 0;
};

template <class T> class shared_ptr_pointer : public shared_count {
  T data_;

public:
  shared_ptr_pointer(T p) : data_(p) {}

private:
  void on_zero_shared() override;

  shared_ptr_pointer(const shared_ptr_pointer &) = delete;
  shared_ptr_pointer &operator=(const shared_ptr_pointer &) = delete;
};

template <class T> void shared_ptr_pointer<T>::on_zero_shared() {
  delete data_;
}

template <class T> class shared_ptr_emplace : public shared_count {
  T data_;

public:
  shared_ptr_emplace() : data_() {}

  template <class A0> shared_ptr_emplace(A0 &a0) : data_(a0) {}

  template <class A0, class A1>
  shared_ptr_emplace(A0 &a0, A1 &a1) : data_(a0, a1) {}

  template <class A0, class A1, class A2>
  shared_ptr_emplace(A0 &a0, A1 &a1, A2 &a2) : data_(a0, a1, a2) {}

  template <class A0, class A1, class A2, class A3>
  shared_ptr_emplace(A0 &a0, A1 &a1, A2 &a2, A3 &a3) : data_(a0, a1, a2, a3) {}

  template <class A0, class A1, class A2, class A3, class A4>
  shared_ptr_emplace(A0 &a0, A1 &a1, A2 &a2, A3 &a3, A4 &a4)
      : data_(a0, a1, a2, a3, a4) {}

private:
  void on_zero_shared() override;

public:
  T *get() { return &data_; }
};

template <class T> void shared_ptr_emplace<T>::on_zero_shared() {}

} // namespace imp

template <class T> class SharingPtr {
public:
  typedef T element_type;

private:
  element_type *ptr_;
  imp::shared_count *cntrl_;

  struct nat {
    int for_bool_;
  };

public:
  SharingPtr();
  SharingPtr(std::nullptr_t);
  template <class Y> explicit SharingPtr(Y *p);
  template <class Y> explicit SharingPtr(Y *p, imp::shared_count *ctrl_block);
  template <class Y> SharingPtr(const SharingPtr<Y> &r, element_type *p);
  SharingPtr(const SharingPtr &r);
  template <class Y> SharingPtr(const SharingPtr<Y> &r);

  ~SharingPtr();

  SharingPtr &operator=(const SharingPtr &r);
  template <class Y> SharingPtr &operator=(const SharingPtr<Y> &r);

  void swap(SharingPtr &r);
  void reset();
  template <class Y> void reset(Y *p);

  element_type *get() const { return ptr_; }
  element_type &operator*() const { return *ptr_; }
  element_type *operator->() const { return ptr_; }
  long use_count() const { return cntrl_ ? cntrl_->use_count() : 0; }
  bool unique() const { return use_count() == 1; }
  bool empty() const { return cntrl_ == nullptr; }
  operator nat *() const { return (nat *)get(); }

  static SharingPtr<T> make_shared();

  template <class A0> static SharingPtr<T> make_shared(A0 &);

  template <class A0, class A1> static SharingPtr<T> make_shared(A0 &, A1 &);

  template <class A0, class A1, class A2>
  static SharingPtr<T> make_shared(A0 &, A1 &, A2 &);

  template <class A0, class A1, class A2, class A3>
  static SharingPtr<T> make_shared(A0 &, A1 &, A2 &, A3 &);

  template <class A0, class A1, class A2, class A3, class A4>
  static SharingPtr<T> make_shared(A0 &, A1 &, A2 &, A3 &, A4 &);

private:
  template <class U> friend class SharingPtr;
};

template <class T>
inline SharingPtr<T>::SharingPtr() : ptr_(nullptr), cntrl_(nullptr) {}

template <class T>
inline SharingPtr<T>::SharingPtr(std::nullptr_t)
    : ptr_(nullptr), cntrl_(nullptr) {}

template <class T>
template <class Y>
SharingPtr<T>::SharingPtr(Y *p) : ptr_(p), cntrl_(nullptr) {
  std::unique_ptr<Y> hold(p);
  typedef imp::shared_ptr_pointer<Y *> _CntrlBlk;
  cntrl_ = new _CntrlBlk(p);
  hold.release();
}

template <class T>
template <class Y>
SharingPtr<T>::SharingPtr(Y *p, imp::shared_count *cntrl_block)
    : ptr_(p), cntrl_(cntrl_block) {}

template <class T>
template <class Y>
inline SharingPtr<T>::SharingPtr(const SharingPtr<Y> &r, element_type *p)
    : ptr_(p), cntrl_(r.cntrl_) {
  if (cntrl_)
    cntrl_->add_shared();
}

template <class T>
inline SharingPtr<T>::SharingPtr(const SharingPtr &r)
    : ptr_(r.ptr_), cntrl_(r.cntrl_) {
  if (cntrl_)
    cntrl_->add_shared();
}

template <class T>
template <class Y>
inline SharingPtr<T>::SharingPtr(const SharingPtr<Y> &r)
    : ptr_(r.ptr_), cntrl_(r.cntrl_) {
  if (cntrl_)
    cntrl_->add_shared();
}

template <class T> SharingPtr<T>::~SharingPtr() {
  if (cntrl_)
    cntrl_->release_shared();
}

template <class T>
inline SharingPtr<T> &SharingPtr<T>::operator=(const SharingPtr &r) {
  SharingPtr(r).swap(*this);
  return *this;
}

template <class T>
template <class Y>
inline SharingPtr<T> &SharingPtr<T>::operator=(const SharingPtr<Y> &r) {
  SharingPtr(r).swap(*this);
  return *this;
}

template <class T> inline void SharingPtr<T>::swap(SharingPtr &r) {
  std::swap(ptr_, r.ptr_);
  std::swap(cntrl_, r.cntrl_);
}

template <class T> inline void SharingPtr<T>::reset() {
  SharingPtr().swap(*this);
}

template <class T> template <class Y> inline void SharingPtr<T>::reset(Y *p) {
  SharingPtr(p).swap(*this);
}

template <class T> SharingPtr<T> SharingPtr<T>::make_shared() {
  typedef imp::shared_ptr_emplace<T> CntrlBlk;
  SharingPtr<T> r;
  r.cntrl_ = new CntrlBlk();
  r.ptr_ = static_cast<CntrlBlk *>(r.cntrl_)->get();
  return r;
}

template <class T>
template <class A0>
SharingPtr<T> SharingPtr<T>::make_shared(A0 &a0) {
  typedef imp::shared_ptr_emplace<T> CntrlBlk;
  SharingPtr<T> r;
  r.cntrl_ = new CntrlBlk(a0);
  r.ptr_ = static_cast<CntrlBlk *>(r.cntrl_)->get();
  return r;
}

template <class T>
template <class A0, class A1>
SharingPtr<T> SharingPtr<T>::make_shared(A0 &a0, A1 &a1) {
  typedef imp::shared_ptr_emplace<T> CntrlBlk;
  SharingPtr<T> r;
  r.cntrl_ = new CntrlBlk(a0, a1);
  r.ptr_ = static_cast<CntrlBlk *>(r.cntrl_)->get();
  return r;
}

template <class T>
template <class A0, class A1, class A2>
SharingPtr<T> SharingPtr<T>::make_shared(A0 &a0, A1 &a1, A2 &a2) {
  typedef imp::shared_ptr_emplace<T> CntrlBlk;
  SharingPtr<T> r;
  r.cntrl_ = new CntrlBlk(a0, a1, a2);
  r.ptr_ = static_cast<CntrlBlk *>(r.cntrl_)->get();
  return r;
}

template <class T>
template <class A0, class A1, class A2, class A3>
SharingPtr<T> SharingPtr<T>::make_shared(A0 &a0, A1 &a1, A2 &a2, A3 &a3) {
  typedef imp::shared_ptr_emplace<T> CntrlBlk;
  SharingPtr<T> r;
  r.cntrl_ = new CntrlBlk(a0, a1, a2, a3);
  r.ptr_ = static_cast<CntrlBlk *>(r.cntrl_)->get();
  return r;
}

template <class T>
template <class A0, class A1, class A2, class A3, class A4>
SharingPtr<T> SharingPtr<T>::make_shared(A0 &a0, A1 &a1, A2 &a2, A3 &a3,
                                         A4 &a4) {
  typedef imp::shared_ptr_emplace<T> CntrlBlk;
  SharingPtr<T> r;
  r.cntrl_ = new CntrlBlk(a0, a1, a2, a3, a4);
  r.ptr_ = static_cast<CntrlBlk *>(r.cntrl_)->get();
  return r;
}

template <class T> inline SharingPtr<T> make_shared() {
  return SharingPtr<T>::make_shared();
}

template <class T, class A0> inline SharingPtr<T> make_shared(A0 &a0) {
  return SharingPtr<T>::make_shared(a0);
}

template <class T, class A0, class A1>
inline SharingPtr<T> make_shared(A0 &a0, A1 &a1) {
  return SharingPtr<T>::make_shared(a0, a1);
}

template <class T, class A0, class A1, class A2>
inline SharingPtr<T> make_shared(A0 &a0, A1 &a1, A2 &a2) {
  return SharingPtr<T>::make_shared(a0, a1, a2);
}

template <class T, class A0, class A1, class A2, class A3>
inline SharingPtr<T> make_shared(A0 &a0, A1 &a1, A2 &a2, A3 &a3) {
  return SharingPtr<T>::make_shared(a0, a1, a2, a3);
}

template <class T, class A0, class A1, class A2, class A3, class A4>
inline SharingPtr<T> make_shared(A0 &a0, A1 &a1, A2 &a2, A3 &a3, A4 &a4) {
  return SharingPtr<T>::make_shared(a0, a1, a2, a3, a4);
}

template <class T, class U>
inline bool operator==(const SharingPtr<T> &__x, const SharingPtr<U> &__y) {
  return __x.get() == __y.get();
}

template <class T, class U>
inline bool operator!=(const SharingPtr<T> &__x, const SharingPtr<U> &__y) {
  return !(__x == __y);
}

template <class T, class U>
inline bool operator<(const SharingPtr<T> &__x, const SharingPtr<U> &__y) {
  return __x.get() < __y.get();
}

template <class T> inline void swap(SharingPtr<T> &__x, SharingPtr<T> &__y) {
  __x.swap(__y);
}

template <class T, class U>
inline SharingPtr<T> static_pointer_cast(const SharingPtr<U> &r) {
  return SharingPtr<T>(r, static_cast<T *>(r.get()));
}

template <class T, class U>
SharingPtr<T> const_pointer_cast(const SharingPtr<U> &r) {
  return SharingPtr<T>(r, const_cast<T *>(r.get()));
}

} // namespace lldb_private

#endif // utility_SharingPtr_h_
