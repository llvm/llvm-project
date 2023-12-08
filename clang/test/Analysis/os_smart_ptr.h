#ifndef _OS_SMART_POINTER_H
#define _OS_SMART_POINTER_H

#include "os_object_base.h"

namespace os {

template<class T>
struct smart_ptr {
  smart_ptr() : pointer(nullptr) {}

  explicit smart_ptr(T *&p) : pointer(p) {
    if (pointer) {
      _retain(pointer);
    }
  }

  smart_ptr(smart_ptr const &rhs) : pointer(rhs.pointer) {
    if (pointer) {
      _retain(pointer);
    }
  }

  smart_ptr & operator=(T *&rhs) {
    smart_ptr(rhs).swap(*this);
    return *this;
  }

  smart_ptr & operator=(smart_ptr &rhs) {
    smart_ptr(rhs).swap(*this);
    return *this;
  }

  ~smart_ptr() {
    if (pointer) {
      _release(pointer);
    }
  }

  void reset() {
    smart_ptr().swap(*this);
  }

  T *get() const {
    return pointer;
  }

  T ** get_for_out_param() {
    reset();
    return &pointer;
  }

  T * operator->() const {
    return pointer;
  }

  explicit
  operator bool() const {
    return pointer != nullptr;
  }

  inline void
  swap(smart_ptr &p) {
    T *temp = pointer;
    pointer = p.pointer;
    p.pointer = temp;
  }

  static inline void
  _retain(T *obj) {
    obj->retain();
  }

  static inline void
  _release(T *obj) {
    obj->release();
  }

  static inline T *
  _alloc() {
    return new T;
  }

  T *pointer;
};
} // namespace os

#endif /* _OS_SMART_POINTER_H */
