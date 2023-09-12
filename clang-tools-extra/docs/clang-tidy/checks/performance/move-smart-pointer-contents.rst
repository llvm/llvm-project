.. title:: clang-tidy - performance-move-smart-pointer-contents

performance-move-smart-pointer-contents
=======================================

Recommends avoiding moving out of a smart pointer when moving the pointer is
cheaper.

Given a smart pointer containing a movable type, such as a
``std::unique_ptr<SomeProtocolBuffer>``, it's possible to move the contents of the
pointer rather than the pointer itself (i.e. ``std::move(*p)`` rather than
``*std::move(p)``). Doing so is a pessimization if the type cannot be efficiently
moved, as the pointer will be quicker than a larger type.

Options
-------

.. option :: UniquePointerClasses

    A semicolon-separated list of class names that should be treated as unique
    pointers. By default only ``std::unique_ptr`` is included.
