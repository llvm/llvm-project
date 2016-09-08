//===-- Mutex.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Mutex_h_
#define liblldb_Mutex_h_

// C Includes
// C++ Includes
#ifdef LLDB_CONFIGURATION_DEBUG
#include <string>
#endif

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-types.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class Mutex Mutex.h "lldb/Host/Mutex.h"
/// @brief A C++ wrapper class for pthread mutexes.
//----------------------------------------------------------------------
class Mutex {
public:
  friend class Locker;
  friend class Condition;

  enum Type {
    eMutexTypeNormal,   ///< Mutex that can't recursively entered by the same
                        ///thread
    eMutexTypeRecursive ///< Mutex can be recursively entered by the same thread
  };

  //------------------------------------------------------------------
  /// @class Mutex::Locker
  ///
  /// A scoped locking class that allows a variety of pthread mutex
  /// objects to have a mutex locked when an Mutex::Locker
  /// object is created, and unlocked when it goes out of scope or
  /// when the Mutex::Locker::Reset(pthread_mutex_t *)
  /// is called. This provides an exception safe way to lock a mutex
  /// in a scope.
  //------------------------------------------------------------------
  class Locker {
  public:
    //--------------------------------------------------------------
    /// Default constructor.
    ///
    /// This will create a scoped mutex locking object that doesn't
    /// have a mutex to lock. One will need to be provided using the
    /// Mutex::Locker::Reset(pthread_mutex_t *) method.
    ///
    /// @see Mutex::Locker::Reset(pthread_mutex_t *)
    //--------------------------------------------------------------
    Locker();

    //--------------------------------------------------------------
    /// Constructor with a Mutex object.
    ///
    /// This will create a scoped mutex locking object that extracts
    /// the mutex owned by \a m and locks it.
    ///
    /// @param[in] m
    ///     An instance of a Mutex object that contains a
    ///     valid mutex object.
    //--------------------------------------------------------------
    Locker(Mutex &m);

    //--------------------------------------------------------------
    /// Constructor with a Mutex object pointer.
    ///
    /// This will create a scoped mutex locking object that extracts
    /// the mutex owned by a m and locks it.
    ///
    /// @param[in] m
    ///     A pointer to instance of a Mutex object that
    ///     contains a valid mutex object.
    //--------------------------------------------------------------
    Locker(Mutex *m);

    //--------------------------------------------------------------
    /// Destructor
    ///
    /// Unlocks any valid pthread_mutex_t that this object may
    /// contain.
    //--------------------------------------------------------------
    ~Locker();

    //--------------------------------------------------------------
    /// Change the contained mutex.
    ///
    /// Unlock the current mutex in this object (if it contains a
    /// valid mutex) and lock the new \a mutex object if it is
    /// non-nullptr.
    //--------------------------------------------------------------
    void Lock(Mutex &mutex);

    void Lock(Mutex *mutex) {
      if (mutex)
        Lock(*mutex);
    }

    //--------------------------------------------------------------
    /// Change the contained mutex only if the mutex can be locked.
    ///
    /// Unlock the current mutex in this object (if it contains a
    /// valid mutex) and try to lock \a mutex. If \a mutex can be
    /// locked this object will take ownership of the lock and will
    /// unlock it when it goes out of scope or Reset or TryLock are
    /// called again. If the mutex is already locked, this object
    /// will not take ownership of the mutex.
    ///
    /// @return
    ///     Returns \b true if the lock was acquired and the this
    ///     object will unlock the mutex when it goes out of scope,
    ///     returns \b false otherwise.
    //--------------------------------------------------------------
    bool TryLock(Mutex &mutex, const char *failure_message = nullptr);

    bool TryLock(Mutex *mutex, const char *failure_message = nullptr) {
      if (mutex)
        return TryLock(*mutex, failure_message);
      else
        return false;
    }

    void Unlock();

  protected:
    //--------------------------------------------------------------
    /// Member variables
    //--------------------------------------------------------------
    Mutex *m_mutex_ptr;

  private:
    Locker(const Locker &);
    const Locker &operator=(const Locker &);
  };

  //------------------------------------------------------------------
  /// Default constructor.
  ///
  /// Creates a pthread mutex with no attributes.
  //------------------------------------------------------------------
  Mutex();

  //------------------------------------------------------------------
  /// Default constructor.
  ///
  /// Creates a pthread mutex with \a type as the mutex type.
  /// Valid values for \a type include:
  ///     @li Mutex::Type::eMutexTypeNormal
  ///     @li Mutex::Type::eMutexTypeRecursive
  ///
  /// @param[in] type
  ///     The type of the mutex.
  ///
  /// @see ::pthread_mutexattr_settype()
  //------------------------------------------------------------------
  Mutex(Mutex::Type type);

//------------------------------------------------------------------
/// Destructor.
///
/// Destroys the mutex owned by this object.
//------------------------------------------------------------------
#ifdef LLDB_CONFIGURATION_DEBUG
  virtual
#endif
      ~Mutex();

//------------------------------------------------------------------
/// Lock the mutex.
///
/// Locks the mutex owned by this object. If the mutex is already
/// locked, the calling thread will block until the mutex becomes
/// available.
///
/// @return
///     The error code from \c pthread_mutex_lock().
//------------------------------------------------------------------
#ifdef LLDB_CONFIGURATION_DEBUG
  virtual
#endif
      int
      Lock();

//------------------------------------------------------------------
/// Try to lock the mutex.
///
/// Attempts to lock the mutex owned by this object without blocking.
/// If the mutex is already locked, TryLock() will not block waiting
/// for the mutex, but will return an error condition.
///
/// @return
///     The error code from \c pthread_mutex_trylock().
//------------------------------------------------------------------
#ifdef LLDB_CONFIGURATION_DEBUG
  virtual
#endif
      int
      TryLock(const char *failure_message = nullptr);

//------------------------------------------------------------------
/// Unlock the mutex.
///
/// If the current thread holds the lock on the owned mutex, then
/// Unlock() will unlock the mutex. Calling Unlock() on this object
/// when the calling thread does not hold the lock will result in
/// undefined behavior.
///
/// @return
///     The error code from \c pthread_mutex_unlock().
//------------------------------------------------------------------
#ifdef LLDB_CONFIGURATION_DEBUG
  virtual
#endif
      int
      Unlock();

protected:
  //------------------------------------------------------------------
  // Member variables
  //------------------------------------------------------------------
  // TODO: Hide the mutex in the implementation file in case we ever need to
  // port to an
  // architecture that doesn't have pthread mutexes.
  lldb::mutex_t m_mutex; ///< The OS mutex object.

private:
  //------------------------------------------------------------------
  /// Mutex get accessor.
  ///
  /// @return
  ///     A pointer to the pthread mutex object owned by this object.
  //------------------------------------------------------------------
  lldb::mutex_t *GetMutex();

  Mutex(const Mutex &);
  const Mutex &operator=(const Mutex &);
};

#ifdef LLDB_CONFIGURATION_DEBUG
class TrackingMutex : public Mutex {
public:
  TrackingMutex() : Mutex() {}
  TrackingMutex(Mutex::Type type) : Mutex(type) {}

  virtual ~TrackingMutex() = default;

  virtual int Unlock();

  virtual int TryLock(const char *failure_message = nullptr) {
    int return_value = Mutex::TryLock();
    if (return_value != 0 && failure_message != nullptr) {
      m_failure_message.assign(failure_message);
      m_thread_that_tried = pthread_self();
    }
    return return_value;
  }

protected:
  pthread_t m_thread_that_tried;
  std::string m_failure_message;
};

class LoggingMutex : public Mutex {
public:
  LoggingMutex() : Mutex(), m_locked(false) {}
  LoggingMutex(Mutex::Type type) : Mutex(type), m_locked(false) {}

  virtual ~LoggingMutex() = default;

  virtual int Lock();

  virtual int Unlock();

  virtual int TryLock(const char *failure_message = nullptr);

protected:
  bool m_locked;
};
#endif // LLDB_CONFIGURATION_DEBUG

} // namespace lldb_private

#endif // liblldb_Mutex_h_
