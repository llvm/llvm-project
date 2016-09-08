//===-- Predicate.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Predicate_h_
#define liblldb_Predicate_h_

// C Includes
#include <stdint.h>
#include <time.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Host/Condition.h"
#include "lldb/Host/Mutex.h"
#include "lldb/lldb-defines.h"

//#define DB_PTHREAD_LOG_EVENTS

//----------------------------------------------------------------------
/// Enumerations for broadcasting.
//----------------------------------------------------------------------
namespace lldb_private {

typedef enum {
  eBroadcastNever,   ///< No broadcast will be sent when the value is modified.
  eBroadcastAlways,  ///< Always send a broadcast when the value is modified.
  eBroadcastOnChange ///< Only broadcast if the value changes when the value is
                     ///modified.
} PredicateBroadcastType;

//----------------------------------------------------------------------
/// @class Predicate Predicate.h "lldb/Host/Predicate.h"
/// @brief A C++ wrapper class for providing threaded access to a value
/// of type T.
///
/// A templatized class that provides multi-threaded access to a value
/// of type T. Threads can efficiently wait for bits within T to be set
/// or reset, or wait for T to be set to be equal/not equal to a
/// specified values.
//----------------------------------------------------------------------
template <class T> class Predicate {
public:
  //------------------------------------------------------------------
  /// Default constructor.
  ///
  /// Initializes the mutex, condition and value with their default
  /// constructors.
  //------------------------------------------------------------------
  Predicate() : m_value(), m_mutex(), m_condition() {}

  //------------------------------------------------------------------
  /// Construct with initial T value \a initial_value.
  ///
  /// Initializes the mutex and condition with their default
  /// constructors, and initializes the value with \a initial_value.
  ///
  /// @param[in] initial_value
  ///     The initial value for our T object.
  //------------------------------------------------------------------
  Predicate(T initial_value)
      : m_value(initial_value), m_mutex(), m_condition() {}

  //------------------------------------------------------------------
  /// Destructor.
  ///
  /// Destroy the condition, mutex, and T objects.
  //------------------------------------------------------------------
  ~Predicate() = default;

  //------------------------------------------------------------------
  /// Value get accessor.
  ///
  /// Copies the current \a m_value in a thread safe manor and returns
  /// the copied value.
  ///
  /// @return
  ///     A copy of the current value.
  //------------------------------------------------------------------
  T GetValue() const {
    Mutex::Locker locker(m_mutex);
    T value = m_value;
    return value;
  }

  //------------------------------------------------------------------
  /// Value set accessor.
  ///
  /// Set the contained \a m_value to \a new_value in a thread safe
  /// way and broadcast if needed.
  ///
  /// @param[in] value
  ///     The new value to set.
  ///
  /// @param[in] broadcast_type
  ///     A value indicating when and if to broadcast. See the
  ///     PredicateBroadcastType enumeration for details.
  ///
  /// @see Predicate::Broadcast()
  //------------------------------------------------------------------
  void SetValue(T value, PredicateBroadcastType broadcast_type) {
    Mutex::Locker locker(m_mutex);
#ifdef DB_PTHREAD_LOG_EVENTS
    printf("%s (value = 0x%8.8x, broadcast_type = %i)\n", __FUNCTION__, value,
           broadcast_type);
#endif
    const T old_value = m_value;
    m_value = value;

    Broadcast(old_value, broadcast_type);
  }

  //------------------------------------------------------------------
  /// Set some bits in \a m_value.
  ///
  /// Logically set the bits \a bits in the contained \a m_value in a
  /// thread safe way and broadcast if needed.
  ///
  /// @param[in] bits
  ///     The bits to set in \a m_value.
  ///
  /// @param[in] broadcast_type
  ///     A value indicating when and if to broadcast. See the
  ///     PredicateBroadcastType enumeration for details.
  ///
  /// @see Predicate::Broadcast()
  //------------------------------------------------------------------
  void SetValueBits(T bits, PredicateBroadcastType broadcast_type) {
    Mutex::Locker locker(m_mutex);
#ifdef DB_PTHREAD_LOG_EVENTS
    printf("%s (bits = 0x%8.8x, broadcast_type = %i)\n", __FUNCTION__, bits,
           broadcast_type);
#endif
    const T old_value = m_value;
    m_value |= bits;

    Broadcast(old_value, broadcast_type);
  }

  //------------------------------------------------------------------
  /// Reset some bits in \a m_value.
  ///
  /// Logically reset (clear) the bits \a bits in the contained
  /// \a m_value in a thread safe way and broadcast if needed.
  ///
  /// @param[in] bits
  ///     The bits to clear in \a m_value.
  ///
  /// @param[in] broadcast_type
  ///     A value indicating when and if to broadcast. See the
  ///     PredicateBroadcastType enumeration for details.
  ///
  /// @see Predicate::Broadcast()
  //------------------------------------------------------------------
  void ResetValueBits(T bits, PredicateBroadcastType broadcast_type) {
    Mutex::Locker locker(m_mutex);
#ifdef DB_PTHREAD_LOG_EVENTS
    printf("%s (bits = 0x%8.8x, broadcast_type = %i)\n", __FUNCTION__, bits,
           broadcast_type);
#endif
    const T old_value = m_value;
    m_value &= ~bits;

    Broadcast(old_value, broadcast_type);
  }

  //------------------------------------------------------------------
  /// Wait for bits to be set in \a m_value.
  ///
  /// Waits in a thread safe way for any bits in \a bits to get
  /// logically set in \a m_value. If any bits are already set in
  /// \a m_value, this function will return without waiting.
  ///
  /// It is possible for the value to be changed between the time
  /// the bits are set and the time the waiting thread wakes up.
  /// If the bits are no longer set when the waiting thread wakes
  /// up, it will go back into a wait state.  It may be necessary
  /// for the calling code to use additional thread synchronization
  /// methods to detect transitory states.
  ///
  /// @param[in] bits
  ///     The bits we are waiting to be set in \a m_value.
  ///
  /// @param[in] abstime
  ///     If non-nullptr, the absolute time at which we should stop
  ///     waiting, else wait an infinite amount of time.
  ///
  /// @return
  ///     Any bits of the requested bits that actually were set within
  ///     the time specified. Zero if a timeout or unrecoverable error
  ///     occurred.
  //------------------------------------------------------------------
  T WaitForSetValueBits(T bits, const TimeValue *abstime = nullptr) {
    int err = 0;
    // pthread_cond_timedwait() or pthread_cond_wait() will atomically
    // unlock the mutex and wait for the condition to be set. When either
    // function returns, they will re-lock the mutex. We use an auto lock/unlock
    // class (Mutex::Locker) to allow us to return at any point in this
    // function and not have to worry about unlocking the mutex.
    Mutex::Locker locker(m_mutex);
#ifdef DB_PTHREAD_LOG_EVENTS
    printf("%s (bits = 0x%8.8x, abstime = %p), m_value = 0x%8.8x\n",
           __FUNCTION__, bits, abstime, m_value);
#endif
    while (err == 0 && ((m_value & bits) == 0)) {
      err = m_condition.Wait(m_mutex, abstime);
    }
#ifdef DB_PTHREAD_LOG_EVENTS
    printf("%s (bits = 0x%8.8x), m_value = 0x%8.8x, returning 0x%8.8x\n",
           __FUNCTION__, bits, m_value, m_value & bits);
#endif

    return m_value & bits;
  }

  //------------------------------------------------------------------
  /// Wait for bits to be reset in \a m_value.
  ///
  /// Waits in a thread safe way for any bits in \a bits to get
  /// logically reset in \a m_value. If all bits are already reset in
  /// \a m_value, this function will return without waiting.
  ///
  /// It is possible for the value to be changed between the time
  /// the bits are reset and the time the waiting thread wakes up.
  /// If the bits are no set when the waiting thread wakes up, it will
  /// go back into a wait state.  It may be necessary for the calling
  /// code to use additional thread synchronization methods to detect
  /// transitory states.
  ///
  /// @param[in] bits
  ///     The bits we are waiting to be reset in \a m_value.
  ///
  /// @param[in] abstime
  ///     If non-nullptr, the absolute time at which we should stop
  ///     waiting, else wait an infinite amount of time.
  ///
  /// @return
  ///     Zero on successful waits, or non-zero if a timeout or
  ///     unrecoverable error occurs.
  //------------------------------------------------------------------
  T WaitForResetValueBits(T bits, const TimeValue *abstime = nullptr) {
    int err = 0;

    // pthread_cond_timedwait() or pthread_cond_wait() will atomically
    // unlock the mutex and wait for the condition to be set. When either
    // function returns, they will re-lock the mutex. We use an auto lock/unlock
    // class (Mutex::Locker) to allow us to return at any point in this
    // function and not have to worry about unlocking the mutex.
    Mutex::Locker locker(m_mutex);

#ifdef DB_PTHREAD_LOG_EVENTS
    printf("%s (bits = 0x%8.8x, abstime = %p), m_value = 0x%8.8x\n",
           __FUNCTION__, bits, abstime, m_value);
#endif
    while (err == 0 && ((m_value & bits) != 0)) {
      err = m_condition.Wait(m_mutex, abstime);
    }

#ifdef DB_PTHREAD_LOG_EVENTS
    printf("%s (bits = 0x%8.8x), m_value = 0x%8.8x, returning 0x%8.8x\n",
           __FUNCTION__, bits, m_value, m_value & bits);
#endif
    return m_value & bits;
  }

  //------------------------------------------------------------------
  /// Wait for \a m_value to be equal to \a value.
  ///
  /// Waits in a thread safe way for \a m_value to be equal to \a
  /// value. If \a m_value is already equal to \a value, this
  /// function will return without waiting.
  ///
  /// It is possible for the value to be changed between the time
  /// the value is set and the time the waiting thread wakes up.
  /// If the value no longer matches the requested value when the
  /// waiting thread wakes up, it will go back into a wait state.  It
  /// may be necessary for the calling code to use additional thread
  /// synchronization methods to detect transitory states.
  ///
  /// @param[in] value
  ///     The value we want \a m_value to be equal to.
  ///
  /// @param[in] abstime
  ///     If non-nullptr, the absolute time at which we should stop
  ///     waiting, else wait an infinite amount of time.
  ///
  /// @param[out] timed_out
  ///     If not null, set to true if we return because of a time out,
  ///     and false if the value was set.
  ///
  /// @return
  ///     @li \b true if the \a m_value is equal to \a value
  ///     @li \b false otherwise
  //------------------------------------------------------------------
  bool WaitForValueEqualTo(T value, const TimeValue *abstime = nullptr,
                           bool *timed_out = nullptr) {
    int err = 0;
    // pthread_cond_timedwait() or pthread_cond_wait() will atomically
    // unlock the mutex and wait for the condition to be set. When either
    // function returns, they will re-lock the mutex. We use an auto lock/unlock
    // class (Mutex::Locker) to allow us to return at any point in this
    // function and not have to worry about unlocking the mutex.
    Mutex::Locker locker(m_mutex);

#ifdef DB_PTHREAD_LOG_EVENTS
    printf("%s (value = 0x%8.8x, abstime = %p), m_value = 0x%8.8x\n",
           __FUNCTION__, value, abstime, m_value);
#endif
    if (timed_out)
      *timed_out = false;

    while (err == 0 && m_value != value) {
      err = m_condition.Wait(m_mutex, abstime, timed_out);
    }

    return m_value == value;
  }

  //------------------------------------------------------------------
  /// Wait for \a m_value to be equal to \a value and then set it to
  /// a new value.
  ///
  /// Waits in a thread safe way for \a m_value to be equal to \a
  /// value and then sets \a m_value to \a new_value. If \a m_value
  /// is already equal to \a value, this function will immediately
  /// set \a m_value to \a new_value and return without waiting.
  ///
  /// It is possible for the value to be changed between the time
  /// the value is set and the time the waiting thread wakes up.
  /// If the value no longer matches the requested value when the
  /// waiting thread wakes up, it will go back into a wait state.  It
  /// may be necessary for the calling code to use additional thread
  /// synchronization methods to detect transitory states.
  ///
  /// @param[in] value
  ///     The value we want \a m_value to be equal to.
  ///
  /// @param[in] new_value
  ///     The value to which \a m_value will be set if \b true is
  ///     returned.
  ///
  /// @param[in] abstime
  ///     If non-nullptr, the absolute time at which we should stop
  ///     waiting, else wait an infinite amount of time.
  ///
  /// @param[out] timed_out
  ///     If not null, set to true if we return because of a time out,
  ///     and false if the value was set.
  ///
  /// @return
  ///     @li \b true if the \a m_value became equal to \a value
  ///     @li \b false otherwise
  //------------------------------------------------------------------
  bool WaitForValueEqualToAndSetValueTo(T wait_value, T new_value,
                                        const TimeValue *abstime = nullptr,
                                        bool *timed_out = nullptr) {
    int err = 0;
    // pthread_cond_timedwait() or pthread_cond_wait() will atomically
    // unlock the mutex and wait for the condition to be set. When either
    // function returns, they will re-lock the mutex. We use an auto lock/unlock
    // class (Mutex::Locker) to allow us to return at any point in this
    // function and not have to worry about unlocking the mutex.
    Mutex::Locker locker(m_mutex);

#ifdef DB_PTHREAD_LOG_EVENTS
    printf("%s (wait_value = 0x%8.8x, new_value = 0x%8.8x, abstime = %p), "
           "m_value = 0x%8.8x\n",
           __FUNCTION__, wait_value, new_value, abstime, m_value);
#endif
    if (timed_out)
      *timed_out = false;

    while (err == 0 && m_value != wait_value) {
      err = m_condition.Wait(m_mutex, abstime, timed_out);
    }

    if (m_value == wait_value) {
      m_value = new_value;
      return true;
    }

    return false;
  }

  //------------------------------------------------------------------
  /// Wait for \a m_value to not be equal to \a value.
  ///
  /// Waits in a thread safe way for \a m_value to not be equal to \a
  /// value. If \a m_value is already not equal to \a value, this
  /// function will return without waiting.
  ///
  /// It is possible for the value to be changed between the time
  /// the value is set and the time the waiting thread wakes up.
  /// If the value is equal to the test value when the waiting thread
  /// wakes up, it will go back into a wait state.  It may be
  /// necessary for the calling code to use additional thread
  /// synchronization methods to detect transitory states.
  ///
  /// @param[in] value
  ///     The value we want \a m_value to not be equal to.
  ///
  /// @param[out] new_value
  ///     The new value if \b true is returned.
  ///
  /// @param[in] abstime
  ///     If non-nullptr, the absolute time at which we should stop
  ///     waiting, else wait an infinite amount of time.
  ///
  /// @return
  ///     @li \b true if the \a m_value is equal to \a value
  ///     @li \b false otherwise
  //------------------------------------------------------------------
  bool WaitForValueNotEqualTo(T value, T &new_value,
                              const TimeValue *abstime = nullptr) {
    int err = 0;
    // pthread_cond_timedwait() or pthread_cond_wait() will atomically
    // unlock the mutex and wait for the condition to be set. When either
    // function returns, they will re-lock the mutex. We use an auto lock/unlock
    // class (Mutex::Locker) to allow us to return at any point in this
    // function and not have to worry about unlocking the mutex.
    Mutex::Locker locker(m_mutex);
#ifdef DB_PTHREAD_LOG_EVENTS
    printf("%s (value = 0x%8.8x, abstime = %p), m_value = 0x%8.8x\n",
           __FUNCTION__, value, abstime, m_value);
#endif
    while (err == 0 && m_value == value) {
      err = m_condition.Wait(m_mutex, abstime);
    }

    if (m_value != value) {
      new_value = m_value;
      return true;
    }
    return false;
  }

protected:
  //----------------------------------------------------------------------
  // pthread condition and mutex variable to control access and allow
  // blocking between the main thread and the spotlight index thread.
  //----------------------------------------------------------------------
  T m_value; ///< The templatized value T that we are protecting access to
  mutable Mutex m_mutex; ///< The mutex to use when accessing the data
  Condition m_condition; ///< The pthread condition variable to use for
                         ///signaling that data available or changed.

private:
  //------------------------------------------------------------------
  /// Broadcast if needed.
  ///
  /// Check to see if we need to broadcast to our condition variable
  /// depending on the \a old_value and on the \a broadcast_type.
  ///
  /// If \a broadcast_type is eBroadcastNever, no broadcast will be
  /// sent.
  ///
  /// If \a broadcast_type is eBroadcastAlways, the condition variable
  /// will always be broadcast.
  ///
  /// If \a broadcast_type is eBroadcastOnChange, the condition
  /// variable be broadcast if the owned value changes.
  //------------------------------------------------------------------
  void Broadcast(T old_value, PredicateBroadcastType broadcast_type) {
    bool broadcast =
        (broadcast_type == eBroadcastAlways) ||
        ((broadcast_type == eBroadcastOnChange) && old_value != m_value);
#ifdef DB_PTHREAD_LOG_EVENTS
    printf("%s (old_value = 0x%8.8x, broadcast_type = %i) m_value = 0x%8.8x, "
           "broadcast = %u\n",
           __FUNCTION__, old_value, broadcast_type, m_value, broadcast);
#endif
    if (broadcast)
      m_condition.Broadcast();
  }

  DISALLOW_COPY_AND_ASSIGN(Predicate);
};

} // namespace lldb_private

#endif // liblldb_Predicate_h_
