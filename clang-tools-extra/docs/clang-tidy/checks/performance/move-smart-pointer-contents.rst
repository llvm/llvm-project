.. title:: clang-tidy - performance-move-smart-pointer-contents

performance-move-smart-pointer-contents
=======================================

Given a smart pointer containing a movable type, such as a
`std::unique_ptr<SomeProtocolBuffer>`, it's possible to move the contents of the
pointer rather than the pointer itself (ie `std::move(*p)` rather than
`*std::move(p)`). Doing so is a pessimization, as if the type could be efficiently
moved we wouldn't need to put it in a `unique_ptr` to begin with.

Options
-------

.. option :: UniquePointerClasses

    A semicolon-separated list of class names that should be treated as unique
    pointers. By default only `std::unique_ptr` is included.

.. option :: SharedPointerClasses

   A semicolon-separated list of class names that should be treated as shared
   pointers. By default only `std::shared_ptr` is included.
