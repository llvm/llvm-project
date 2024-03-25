.. title:: clang-tidy - bugprone-unused-local-non-trivial-variable

bugprone-unused-local-non-trivial-variable
==========================================

Warns when a local non trivial variable is unused within a function.
The following types of variables are excluded from this check:

* trivial and trivially copyable
* references and pointers
* exception variables in catch clauses
* static or thread local
* structured bindings
* variables with ``[[maybe_unused]]`` attribute

This check can be configured to warn on all non-trivial variables by setting
`IncludeTypes` to `.*`, and excluding specific types using `ExcludeTypes`.

In the this example, `my_lock` would generate a warning that it is unused.

.. code-block:: c++

   std::mutex my_lock;
   // my_lock local variable is never used

In the next example, `future2` would generate a warning that it is unused.

.. code-block:: c++

   std::future<MyObject> future1;
   std::future<MyObject> future2;
   // ...
   MyObject foo = future1.get();
   // future2 is not used.

Options
-------

.. option:: IncludeTypes

   Semicolon-separated list of regular expressions matching types of variables
   to check. By default the following types are checked:

   * `::std::.*mutex`
   * `::std::future`
   * `::std::basic_string`
   * `::std::basic_regex`
   * `::std::basic_istringstream`
   * `::std::basic_stringstream`
   * `::std::bitset`
   * `::std::filesystem::path`

.. option:: ExcludeTypes

   A semicolon-separated list of regular expressions matching types that are
   excluded from the `IncludeTypes` matches. By default it is an empty list.
