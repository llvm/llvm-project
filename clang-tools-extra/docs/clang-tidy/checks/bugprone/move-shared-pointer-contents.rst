.. title:: clang-tidy - bugprone-move-shared-pointer-contents

bugprone-move-shared-pointer-contents
=====================================


Detects calls to move the contents out of a ``std::shared_ptr`` rather
than moving the pointer itself. In other words, calling
``std::move(*p)`` or ``std::move(*p.get())`` or similar calls with
``std::forward``. Other reference holders may not be expecting the
move and suddenly getting empty or otherwise indeterminate states can
cause issues. Only applies to C++11 and above, as that's when
``std::shared_ptr`` was introduced.

Options
-------
.. option :: SharedPointerClasses

   A semicolon-separated list of class names that should be treated as
   shared pointers. Classes are resolved through aliases, so any alias
   to the defined classes will be considered. Default is
   `::std::shared_ptr;::boost::shared_pointer`.
