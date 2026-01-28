.. title:: clang-tidy - modernize-use-string-view

modernize-use-string-view
=========================

Looks for functions returning ``std::[w|u8|u16|u32]string`` and suggests to
change it to ``std::[...]string_view`` for performance reasons if possible.

Each time a new ``std::string`` is created from a literal, a copy of that
literal is allocated either in ``std::string``'s internal buffer
(for short literals) or on the heap.

For the cases where ``std::string`` is returned from a function,
such allocations can sometimes be eliminated by using ``std::string_view``
as a return type.

This check looks for such functions returning ``std::string`` constructed from
the literals and suggests replacing their return type to ``std::string_view``.

It handles ``std::string``, ``std::wstring``, ``std::u8string``,
``std::u16string`` and ``std::u32string`` along with their aliases and selects
the proper kind of ``std::string_view`` to return.

Consider the following example:

.. code-block:: c++

    std::string foo(int i) {
      switch(i) {
        case 1:
          return "case 1";
        ...
        default:
          return "default";
      }
    }

In the code above a new ``std::string`` object is created on each function
invocation, making a copy of a string literal and possibly allocating a memory
on the heap.

The check gets this code transformed into:

.. code-block:: c++

    std::string_view foo(int i) {
      switch(i) {
        case 1:
          return "case 1";
        ...
        default:
          return "default";
      }
    }

New version re-uses statically allocated literals without additional overhead.

Suppressing diagnostic
----------------------

To prevent an undesired diagnostic wrap the string literal with an explicit
``std::string(...)`` constructor:

.. code-block:: c++

    std::string foo() {
      return "default"; //warning and fix are generated
    }

    std::string bar() {
      return std::string("default"); //warning and fix are NOT generated
    }

Limitations
-----------

* No warning and/or fix are generated as for now for these code patterns:

  * ``return std::string("literal");``
  * ``return std::string{"literal"};``
  * ``return "simpleLiteral"s;``
  * ``auto foo() { return "autoReturn"; }``
  * ``auto Trailing() -> std::string { return "Trailing"; }`` warns, doesn't fix
  * returnings from lambda
  * complicated macro and templated code

In some cases the fixed code will not compile due to lack of conversion from
``std::string_view`` to ``std::string``. It can be fixed (preferably) by
converting receiver ``std::string`` to ``std::string_view`` if possible or
simply make an explicit conversion.

.. code-block:: c++

    string foo() {        // <--- will be replaced with string_view
      return "foo";
    }

    void bar() {
      string err = foo(); // <----- error: no viable conversion from
                          // 'std::string_view' (aka 'basic_string_view<char>')
                          // to 'std::string' (aka 'basic_string<char>')

      string fix(foo());  // <----- no errors
    }


Options
-------

.. option:: IgnoredFunctions

   A semicolon-separated list of the names of functions or methods to be
   ignored. Regular expressions are accepted, e.g. `[Rr]ef(erence)?$` matches
   every type with suffix ``Ref``, ``ref``, ``Reference`` and ``reference``.

   If a name in the list contains the sequence `::` it is matched against the
   qualified type name (i.e. ``namespace::Type``), otherwise it is matched
   against only the type name (i.e. ``Type``).

   The default is `toString$;ToString$;to_string$`.


.. option:: ReplacementStringViewClass

   A semicolon-separated list of `string=string_view` pairs for replacing
   ``string`` to ``string_view`` counterparts.

   ============= ======================== =======================
   Key           Value (example)          Default value
   ============= ======================== =======================
   ``string``    ``llvm::StringRef``      ``std::string_view``
   ``wstring``   ``boost::wstring_view``  ``std::wstring_view``
   ``u8string``  ``absl::u8string_view``  ``std::u8string_view``
   ``u16string`` ``QStringView``          ``std::u16string_view``
   ``u32string`` ``std::u32zstring_view`` ``std::u32string_view``
   ============= ======================== =======================
