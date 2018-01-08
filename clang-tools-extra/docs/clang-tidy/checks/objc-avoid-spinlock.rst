.. title:: clang-tidy - objc-avoid-spinlock

objc-avoid-spinlock
===================

Finds usages of ``OSSpinlock``, which is deprecated due to potential livelock
problems. 

This check will detect following function invocations:

- ``OSSpinlockLock``
- ``OSSpinlockTry``
- ``OSSpinlockUnlock``

The corresponding information about the problem of ``OSSpinlock``: https://blog.postmates.com/why-spinlocks-are-bad-on-ios-b69fc5221058
