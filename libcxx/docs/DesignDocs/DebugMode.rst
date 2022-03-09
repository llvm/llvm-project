==========
Debug Mode
==========

.. contents::
   :local:

.. _using-debug-mode:

Using the debug mode
====================

Libc++ provides a debug mode that enables special debugging checks meant to detect
incorrect usage of the standard library. These checks are disabled by default, but
they can be enabled using the ``_LIBCPP_DEBUG`` macro.

Note that using the debug mode discussed in this document requires that the library
has been compiled with support for the debug mode (see ``LIBCXX_ENABLE_DEBUG_MODE_SUPPORT``).

Also note that while the debug mode has no effect on libc++'s ABI, it does have broad ODR
implications. Users should compile their whole program at the same debugging level.

The various levels of checking provided by the debug mode follow.

No debugging checks (``_LIBCPP_DEBUG`` not defined)
---------------------------------------------------
When ``_LIBCPP_DEBUG`` is not defined, there are no debugging checks performed by
the library. This is the default.

Comparator consistency checks (``_LIBCPP_DEBUG == 1``)
------------------------------------------------------
Libc++ provides some checks for the consistency of comparators passed to algorithms. Specifically,
many algorithms such as ``binary_search``, ``merge``, ``next_permutation``, and ``sort``, wrap the
user-provided comparator to assert that `!comp(y, x)` whenever `comp(x, y)`. This can cause the
user-provided comparator to be evaluated up to twice as many times as it would be without the
debug mode, and causes the library to violate some of the Standard's complexity clauses.

Iterator debugging checks (``_LIBCPP_DEBUG == 1``)
--------------------------------------------------
Defining ``_LIBCPP_DEBUG`` to ``1`` enables "iterator debugging", which provides
additional assertions about the validity of iterators used by the program.

The following containers and classes support iterator debugging:

- ``std::string``
- ``std::vector<T>`` (``T != bool``)
- ``std::list``
- ``std::unordered_map``
- ``std::unordered_multimap``
- ``std::unordered_set``
- ``std::unordered_multiset``

The remaining containers do not currently support iterator debugging.
Patches welcome.

Randomizing Unspecified Behavior (``_LIBCPP_DEBUG == 1``)
---------------------------------------------------------
This also enables the randomization of unspecified behavior, for
example, for equal elements in ``std::sort`` or randomizing both parts of
the partition after ``std::nth_element`` call. This effort helps you to migrate
to potential future faster versions of these algorithms and deflake your tests
which depend on such behavior. To fix the seed, use
``_LIBCPP_DEBUG_RANDOMIZE_UNSPECIFIED_STABILITY_SEED=seed`` definition.

Handling Assertion Failures
===========================
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
