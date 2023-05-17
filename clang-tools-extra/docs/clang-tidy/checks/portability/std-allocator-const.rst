.. title:: clang-tidy - portability-std-allocator-const

portability-std-allocator-const
===============================

Report use of ``std::vector<const T>`` (and similar containers of const
elements). These are not allowed in standard C++, and should usually be
``std::vector<T>`` instead."

Per C++ ``[allocator.requirements.general]``: "T is any cv-unqualified object
type", ``std::allocator<const T>`` is undefined. Many standard containers use
``std::allocator`` by default and therefore their ``const T`` instantiations are
undefined.

libc++ defines ``std::allocator<const T>`` as an extension which will be removed
in the future.

libstdc++ and MSVC do not support ``std::allocator<const T>``:

.. code:: c++

  // libstdc++ has a better diagnostic since https://gcc.gnu.org/bugzilla/show_bug.cgi?id=48101
  std::deque<const int> deque; // error: static assertion failed: std::deque must have a non-const, non-volatile value_type
  std::set<const int> set; // error: static assertion failed: std::set must have a non-const, non-volatile value_type
  std::vector<int* const> vector; // error: static assertion failed: std::vector must have a non-const, non-volatile value_type

  // MSVC
  // error C2338: static_assert failed: 'The C++ Standard forbids containers of const elements because allocator<const T> is ill-formed.'

Code bases only compiled with libc++ may accrue such undefined usage. This
check finds such code and prevents backsliding while clean-up is ongoing.
