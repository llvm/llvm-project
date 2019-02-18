// Macros to enable testing of both variants of the thread safety analysis.

#if USE_CAPABILITY
#define LOCKABLE                        __attribute__((capability("mutex")))
#define ASSERT_EXCLUSIVE_LOCK(...)      __attribute__((assert_capability(__VA_ARGS__)))
#define ASSERT_SHARED_LOCK(...)         __attribute__((assert_shared_capability(__VA_ARGS__)))
#define EXCLUSIVE_LOCK_FUNCTION(...)    __attribute__((acquire_capability(__VA_ARGS__)))
#define SHARED_LOCK_FUNCTION(...)       __attribute__((acquire_shared_capability(__VA_ARGS__)))
#define EXCLUSIVE_TRYLOCK_FUNCTION(...) __attribute__((try_acquire_capability(__VA_ARGS__)))
#define SHARED_TRYLOCK_FUNCTION(...)    __attribute__((try_acquire_shared_capability(__VA_ARGS__)))
#define EXCLUSIVE_LOCKS_REQUIRED(...)   __attribute__((requires_capability(__VA_ARGS__)))
#define SHARED_LOCKS_REQUIRED(...)      __attribute__((requires_shared_capability(__VA_ARGS__)))
#else
#define LOCKABLE                        __attribute__((lockable))
#define ASSERT_EXCLUSIVE_LOCK(...)      __attribute__((assert_exclusive_lock(__VA_ARGS__)))
#define ASSERT_SHARED_LOCK(...)         __attribute__((assert_shared_lock(__VA_ARGS__)))
#define EXCLUSIVE_LOCK_FUNCTION(...)    __attribute__((exclusive_lock_function(__VA_ARGS__)))
#define SHARED_LOCK_FUNCTION(...)       __attribute__((shared_lock_function(__VA_ARGS__)))
#define EXCLUSIVE_TRYLOCK_FUNCTION(...) __attribute__((exclusive_trylock_function(__VA_ARGS__)))
#define SHARED_TRYLOCK_FUNCTION(...)    __attribute__((shared_trylock_function(__VA_ARGS__)))
#define EXCLUSIVE_LOCKS_REQUIRED(...)   __attribute__((exclusive_locks_required(__VA_ARGS__)))
#define SHARED_LOCKS_REQUIRED(...)      __attribute__((shared_locks_required(__VA_ARGS__)))
#endif

// Lock semantics only
#define UNLOCK_FUNCTION(...)            __attribute__((unlock_function(__VA_ARGS__)))
#define GUARDED_VAR                     __attribute__((guarded_var))
#define PT_GUARDED_VAR                  __attribute__((pt_guarded_var))

// Capabilities only
#define EXCLUSIVE_UNLOCK_FUNCTION(...)  __attribute__((release_capability(__VA_ARGS__)))
#define SHARED_UNLOCK_FUNCTION(...)     __attribute__((release_shared_capability(__VA_ARGS__)))
#define GUARDED_BY(x)                   __attribute__((guarded_by(x)))
#define PT_GUARDED_BY(x)                __attribute__((pt_guarded_by(x)))

// Common
#define SCOPED_LOCKABLE                 __attribute__((scoped_lockable))
#define ACQUIRED_AFTER(...)             __attribute__((acquired_after(__VA_ARGS__)))
#define ACQUIRED_BEFORE(...)            __attribute__((acquired_before(__VA_ARGS__)))
#define LOCK_RETURNED(x)                __attribute__((lock_returned(x)))
#define LOCKS_EXCLUDED(...)             __attribute__((locks_excluded(__VA_ARGS__)))
#define NO_THREAD_SAFETY_ANALYSIS       __attribute__((no_thread_safety_analysis))
