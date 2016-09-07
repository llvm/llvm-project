//===-- CleanUp.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CleanUp_h_
#define liblldb_CleanUp_h_

#include "lldb/lldb-public.h"
#include <functional>

namespace lldb_utility {

//----------------------------------------------------------------------
// Templated class that guarantees that a cleanup callback function will
// be called. The cleanup function will be called once under the
// following conditions:
// - when the object goes out of scope
// - when the user explicitly calls clean.
// - the current value will be cleaned up when a new value is set using
//   set(T value) as long as the current value hasn't already been cleaned.
//
// This class is designed to be used with simple types for type T (like
// file descriptors, opaque handles, pointers, etc). If more complex
// type T objects are desired, we need to probably specialize this class
// to take "const T&" for all input T parameters. Yet if a type T is
// complex already it might be better to build the cleanup functionality
// into T.
//
// The cleanup function must take one argument that is of type T.
// The calback function return type is R. The return value is currently
// needed for "CallbackType". If there is an easy way to get around the
// need for the return value we can change this class.
//
// The two template parameters are:
//    T - The variable type of value that will be stored and used as the
//      sole argument for the cleanup callback.
//    R - The return type for the cleanup function.
//
// EXAMPLES
//  // Use with file handles that get opened where you want to close
//  // them. Below we use "int open(const char *path, int oflag, ...)"
//  // which returns an integer file descriptor. -1 is the invalid file
//  // descriptor so to make an object that will call "int close(int fd)"
//  // automatically we can use:
//
//  CleanUp <int, int> fd(open("/tmp/a.txt", O_RDONLY, 0), -1, close);
//
//  // malloc/free example
//  CleanUp <void *, void> malloced_bytes(malloc(32), NULL, free);
//----------------------------------------------------------------------
template <typename T, typename R = void> class CleanUp {
public:
  typedef T value_type;
  typedef std::function<R(value_type)> CallbackType;

  //----------------------------------------------------------------------
  // Constructor that sets the current value only. No values are
  // considered to be invalid and the cleanup function will be called
  // regardless of the value of m_current_value.
  //----------------------------------------------------------------------
  CleanUp(value_type value, CallbackType callback)
      : m_current_value(value), m_invalid_value(), m_callback(callback),
        m_callback_called(false), m_invalid_value_is_valid(false) {}

  //----------------------------------------------------------------------
  // Constructor that sets the current value and also the invalid value.
  // The cleanup function will be called on "m_value" as long as it isn't
  // equal to "m_invalid_value".
  //----------------------------------------------------------------------
  CleanUp(value_type value, value_type invalid, CallbackType callback)
      : m_current_value(value), m_invalid_value(invalid), m_callback(callback),
        m_callback_called(false), m_invalid_value_is_valid(true) {}

  //----------------------------------------------------------------------
  // Automatically cleanup when this object goes out of scope.
  //----------------------------------------------------------------------
  ~CleanUp() { clean(); }

  //----------------------------------------------------------------------
  // Access the value stored in this class
  //----------------------------------------------------------------------
  value_type get() { return m_current_value; }

  //----------------------------------------------------------------------
  // Access the value stored in this class
  //----------------------------------------------------------------------
  const value_type get() const { return m_current_value; }

  //----------------------------------------------------------------------
  // Reset the owned value to "value". If a current value is valid and
  // the cleanup callback hasn't been called, the previous value will
  // be cleaned up (see void CleanUp::clean()).
  //----------------------------------------------------------------------
  void set(const value_type value) {
    // Cleanup the current value if needed
    clean();
    // Now set the new value and mark our callback as not called
    m_callback_called = false;
    m_current_value = value;
  }

  //----------------------------------------------------------------------
  // Checks is "m_current_value" is valid. The value is considered valid
  // no invalid value was supplied during construction of this object or
  // if an invalid value was supplied and "m_current_value" is not equal
  // to "m_invalid_value".
  //
  // Returns true if "m_current_value" is valid, false otherwise.
  //----------------------------------------------------------------------
  bool is_valid() const {
    if (m_invalid_value_is_valid)
      return m_current_value != m_invalid_value;
    return true;
  }

  //----------------------------------------------------------------------
  // This function will call the cleanup callback provided in the
  // constructor one time if the value is considered valid (See is_valid()).
  // This function sets m_callback_called to true so we don't call the
  // cleanup callback multiple times on the same value.
  //----------------------------------------------------------------------
  void clean() {
    if (m_callback && !m_callback_called) {
      m_callback_called = true;
      if (is_valid())
        m_callback(m_current_value);
    }
  }

  //----------------------------------------------------------------------
  // Cancels the cleanup that would have been called on "m_current_value"
  // if it was valid. This function can be used to release the value
  // contained in this object so ownership can be transferred to the caller.
  //----------------------------------------------------------------------
  value_type release() {
    m_callback_called = true;
    return m_current_value;
  }

private:
  value_type m_current_value;
  const value_type m_invalid_value;
  CallbackType m_callback;
  bool m_callback_called;
  bool m_invalid_value_is_valid;

  // Outlaw default constructor, copy constructor and the assignment operator
  DISALLOW_COPY_AND_ASSIGN(CleanUp);
};

template <typename T, typename R, typename A0> class CleanUp2 {
public:
  typedef T value_type;
  typedef std::function<R(value_type, A0)> CallbackType;

  //----------------------------------------------------------------------
  // Constructor that sets the current value only. No values are
  // considered to be invalid and the cleanup function will be called
  // regardless of the value of m_current_value.
  //----------------------------------------------------------------------
  CleanUp2(value_type value, CallbackType callback, A0 arg)
      : m_current_value(value), m_invalid_value(), m_callback(callback),
        m_callback_called(false), m_invalid_value_is_valid(false),
        m_argument(arg) {}

  //----------------------------------------------------------------------
  // Constructor that sets the current value and also the invalid value.
  // The cleanup function will be called on "m_value" as long as it isn't
  // equal to "m_invalid_value".
  //----------------------------------------------------------------------
  CleanUp2(value_type value, value_type invalid, CallbackType callback, A0 arg)
      : m_current_value(value), m_invalid_value(invalid), m_callback(callback),
        m_callback_called(false), m_invalid_value_is_valid(true),
        m_argument(arg) {}

  //----------------------------------------------------------------------
  // Automatically cleanup when this object goes out of scope.
  //----------------------------------------------------------------------
  ~CleanUp2() { clean(); }

  //----------------------------------------------------------------------
  // Access the value stored in this class
  //----------------------------------------------------------------------
  value_type get() { return m_current_value; }

  //----------------------------------------------------------------------
  // Access the value stored in this class
  //----------------------------------------------------------------------
  const value_type get() const { return m_current_value; }

  //----------------------------------------------------------------------
  // Reset the owned value to "value". If a current value is valid and
  // the cleanup callback hasn't been called, the previous value will
  // be cleaned up (see void CleanUp::clean()).
  //----------------------------------------------------------------------
  void set(const value_type value) {
    // Cleanup the current value if needed
    clean();
    // Now set the new value and mark our callback as not called
    m_callback_called = false;
    m_current_value = value;
  }

  //----------------------------------------------------------------------
  // Checks is "m_current_value" is valid. The value is considered valid
  // no invalid value was supplied during construction of this object or
  // if an invalid value was supplied and "m_current_value" is not equal
  // to "m_invalid_value".
  //
  // Returns true if "m_current_value" is valid, false otherwise.
  //----------------------------------------------------------------------
  bool is_valid() const {
    if (m_invalid_value_is_valid)
      return m_current_value != m_invalid_value;
    return true;
  }

  //----------------------------------------------------------------------
  // This function will call the cleanup callback provided in the
  // constructor one time if the value is considered valid (See is_valid()).
  // This function sets m_callback_called to true so we don't call the
  // cleanup callback multiple times on the same value.
  //----------------------------------------------------------------------
  void clean() {
    if (m_callback && !m_callback_called) {
      m_callback_called = true;
      if (is_valid())
        m_callback(m_current_value, m_argument);
    }
  }

  //----------------------------------------------------------------------
  // Cancels the cleanup that would have been called on "m_current_value"
  // if it was valid. This function can be used to release the value
  // contained in this object so ownership can be transferred to the caller.
  //----------------------------------------------------------------------
  value_type release() {
    m_callback_called = true;
    return m_current_value;
  }

private:
  value_type m_current_value;
  const value_type m_invalid_value;
  CallbackType m_callback;
  bool m_callback_called;
  bool m_invalid_value_is_valid;
  A0 m_argument;

  // Outlaw default constructor, copy constructor and the assignment operator
  DISALLOW_COPY_AND_ASSIGN(CleanUp2);
};

} // namespace lldb_utility

#endif // #ifndef liblldb_CleanUp_h_
