==========
Debug Mode
==========

.. contents::
   :local:

.. _using-debug-mode:

Using Debug Mode
================

Libc++ provides a debug mode that enables assertions meant to detect incorrect
usage of the standard library. By default these assertions are disabled but
they can be enabled using the ``_LIBCPP_DEBUG`` macro.

**_LIBCPP_DEBUG** Macro
-----------------------

**_LIBCPP_DEBUG**:
  This macro is used to enable assertions and iterator debugging checks within
  libc++. By default it is undefined.

  **Values**: ``0``, ``1``

  Defining ``_LIBCPP_DEBUG`` to ``0`` or greater enables most of libc++'s
  assertions. Defining ``_LIBCPP_DEBUG`` to ``1`` enables "iterator debugging"
  which provides additional assertions about the validity of iterators used by
  the program.

  Note that this option has no effect on libc++'s ABI; but it does have broad
  ODR implications. Users should compile their whole program at the same
  debugging level.

Handling Assertion Failures
---------------------------

When a debug assertion fails the assertion handler is called via the
``std::__libcpp_debug_function`` function pointer. It is possible to override
this function pointer using a different handler function. Libc++ provides a
the default handler, ``std::__libcpp_abort_debug_handler``, which aborts the
program. The handler may not return. Libc++ can be changed to use a custom
assertion handler as follows.

.. code-block:: cpp

  #define _LIBCPP_DEBUG 1
  #include <string>
  void my_handler(std::__libcpp_debug_info const&);
  int main(int, char**) {
    std::__libcpp_debug_function = &my_handler;

    std::string::iterator bad_it;
    std::string str("hello world");
    str.insert(bad_it, '!'); // causes debug assertion
    // control flow doesn't return
  }

Debug Mode Checks
=================

Libc++'s debug mode offers two levels of checking. The first enables various
precondition checks throughout libc++. The second additionally enables
"iterator debugging" which checks the validity of iterators used by the program.

Basic Checks
============

These checks are enabled when ``_LIBCPP_DEBUG`` is defined to either 0 or 1.

The following checks are enabled by ``_LIBCPP_DEBUG``:

  * Many algorithms, such as ``binary_search``, ``merge``, ``next_permutation``, and ``sort``,
    wrap the user-provided comparator to assert that `!comp(y, x)` whenever
    `comp(x, y)`. This can cause the user-provided comparator to be evaluated
    up to twice as many times as it would be without ``_LIBCPP_DEBUG``, and
    causes the library to violate some of the Standard's complexity clauses.

  * FIXME: Update this list

Iterator Debugging Checks
=========================

These checks are enabled when ``_LIBCPP_DEBUG`` is defined to 1.

The following containers and STL classes support iterator debugging:

  * ``std::string``
  * ``std::vector<T>`` (``T != bool``)
  * ``std::list``
  * ``std::unordered_map``
  * ``std::unordered_multimap``
  * ``std::unordered_set``
  * ``std::unordered_multiset``

The remaining containers do not currently support iterator debugging.
Patches welcome.
