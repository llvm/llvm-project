//===-- ReadWriteLock.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ReadWriteLock_h_
#define liblldb_ReadWriteLock_h_
#if defined(__cplusplus)

#include "lldb/Host/Mutex.h"
#include "lldb/Host/Condition.h"
#include <pthread.h>
#include <stdint.h>
#include <time.h>

//----------------------------------------------------------------------
/// Enumerations for broadcasting.
//----------------------------------------------------------------------
namespace lldb_private {

//----------------------------------------------------------------------
/// @class ReadWriteLock ReadWriteLock.h "lldb/Host/ReadWriteLock.h"
/// @brief A C++ wrapper class for providing threaded access to a value
/// of type T.
///
/// A templatized class that provides multi-threaded access to a value
/// of type T. Threads can efficiently wait for bits within T to be set
/// or reset, or wait for T to be set to be equal/not equal to a
/// specified values.
//----------------------------------------------------------------------
    
class ReadWriteLock
{
public:
    ReadWriteLock () :
        m_rwlock()
    {
        int err = ::pthread_rwlock_init(&m_rwlock, NULL); (void)err;
//#if LLDB_CONFIGURATION_DEBUG
//        assert(err == 0);
//#endif
    }

    ~ReadWriteLock ()
    {
        int err = ::pthread_rwlock_destroy (&m_rwlock); (void)err;
//#if LLDB_CONFIGURATION_DEBUG
//        assert(err == 0);
//#endif
    }

    bool
    ReadLock ()
    {
        return ::pthread_rwlock_rdlock (&m_rwlock) == 0;
    }

    bool
    ReadTryLock ()
    {
        return ::pthread_rwlock_tryrdlock (&m_rwlock) == 0;
    }

    bool
    ReadUnlock ()
    {
        return ::pthread_rwlock_unlock (&m_rwlock) == 0;
    }
    
    bool
    WriteLock()
    {
        return ::pthread_rwlock_wrlock (&m_rwlock) == 0;
    }
    
    bool
    WriteTryLock()
    {
        return ::pthread_rwlock_trywrlock (&m_rwlock) == 0;
    }
    
    bool
    WriteUnlock ()
    {
        return ::pthread_rwlock_unlock (&m_rwlock) == 0;
    }

    class ReadLocker
    {
    public:
        ReadLocker () :
            m_lock (NULL)
        {
        }

        ReadLocker (ReadWriteLock &lock) :
            m_lock (NULL)
        {
            Lock(&lock);
        }
    

        ReadLocker (ReadWriteLock *lock) :
            m_lock (NULL)
        {
            Lock(lock);
        }
        
        ~ReadLocker()
        {
            Unlock();
        }

        void
        Lock (ReadWriteLock *lock)
        {
            if (m_lock)
            {
                if (m_lock == lock)
                    return; // We already have this lock locked
                else
                    Unlock();
            }
            if (lock)
            {
                lock->ReadLock();
                m_lock = lock;
            }
        }

        // Try to lock the read lock, but only do so if there are no writers.
        bool
        TryLock (ReadWriteLock *lock)
        {
            if (m_lock)
            {
                if (m_lock == lock)
                    return true; // We already have this lock locked
                else
                    Unlock();
            }
            if (lock)
            {
                if (lock->ReadTryLock())
                {
                    m_lock = lock;
                    return true;
                }
            }
            return false;
        }

        void
        Unlock ()
        {
            if (m_lock)
            {
                m_lock->ReadUnlock();
                m_lock = NULL;
            }
        }
        
    protected:
        ReadWriteLock *m_lock;
    private:
        DISALLOW_COPY_AND_ASSIGN(ReadLocker);
    };

    class WriteLocker
    {
    public:
        WriteLocker () :
            m_lock (NULL)
        {
        }

        WriteLocker (ReadWriteLock &lock) :
            m_lock (NULL)
        {
            Lock(&lock);
        }

        WriteLocker (ReadWriteLock *lock) :
            m_lock (NULL)
        {
            Lock(lock);
        }

        ~WriteLocker()
        {
            Unlock();
        }

        void
        Lock (ReadWriteLock *lock)
        {
            if (m_lock)
            {
                if (m_lock == lock)
                    return; // We already have this lock locked
                else
                    Unlock();
            }
            if (lock)
            {
                lock->WriteLock();
                m_lock = lock;
            }
        }

        void
        Unlock ()
        {
            if (m_lock)
            {
                m_lock->WriteUnlock();
                m_lock = NULL;
            }
        }
        
    protected:
        ReadWriteLock *m_lock;
    private:
        DISALLOW_COPY_AND_ASSIGN(WriteLocker);
    };

protected:
    pthread_rwlock_t m_rwlock;
private:
    DISALLOW_COPY_AND_ASSIGN(ReadWriteLock);
};

} // namespace lldb_private

#endif  // #if defined(__cplusplus)
#endif // #ifndef liblldb_ReadWriteLock_h_
