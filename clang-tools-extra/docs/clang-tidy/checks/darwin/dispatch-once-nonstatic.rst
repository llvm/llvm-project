.. title:: clang-tidy - darwin-dispatch-once-nonstatic

darwin-dispatch-once-nonstatic
==============================

Finds declarations of ``dispatch_once_t`` variables without static or global
storage. The behavior of using ``dispatch_once_t`` predicates with automatic or
dynamic storage is undefined by libdispatch, and should be avoided.

It is a common pattern to have functions initialize internal static or global
data once when the function runs, but programmers have been known to miss the
static on the ``dispatch_once_t`` predicate, leading to an uninitialized flag
value at the mercy of the stack.

Programmers have also been known to make ``dispatch_once_t`` variables be
members of structs or classes, with the intent to lazily perform some expensive
struct or class member initialization only once; however, this violates the
libdispatch requirements.

See the discussion section of
`Apple's dispatch_once documentation <https://developer.apple.com/documentation/dispatch/1447169-dispatch_once>`_
for more information.
