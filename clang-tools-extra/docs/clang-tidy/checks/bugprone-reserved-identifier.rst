.. title:: clang-tidy - bugprone-reserved-identifier

bugprone-reserved-identifier
============================

`cert-dcl37-c` and `cert-dcl51-cpp` redirect here as an alias for this check.

Checks for usages of identifiers reserved for use by the implementation.

The C and C++ standards both reserve the following names for such use:

- identifiers that begin with an underscore followed by an uppercase letter;
- identifiers in the global namespace that begin with an underscore.

The C standard additionally reserves names beginning with a double underscore,
while the C++ standard strengthens this to reserve names with a double
underscore occurring anywhere.

Violating the naming rules above results in undefined behavior.

.. code-block:: c++

  namespace NS {
    void __f(); // name is not allowed in user code
    using _Int = int; // same with this
    #define cool__macro // also this
  }
  int _g(); // disallowed in global namespace only

The check can also be inverted, i.e. it can be configured to flag any
identifier that is _not_ a reserved identifier. This mode is for use by e.g.
standard library implementors, to ensure they don't infringe on the user
namespace.

This check does not (yet) check for other reserved names, e.g. macro names
identical to language keywords, and names specifically reserved by language
standards, e.g. C++ 'zombie names' and C future library directions.

This check corresponds to CERT C Coding Standard rule `DCL37-C. Do not declare
or define a reserved identifier
<https://wiki.sei.cmu.edu/confluence/display/c/DCL37-C.+Do+not+declare+or+define+a+reserved+identifier>`_
as well as its C++ counterpart, `DCL51-CPP. Do not declare or define a reserved
identifier
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/DCL51-CPP.+Do+not+declare+or+define+a+reserved+identifier>`_.

Options
-------

.. option:: Invert

   If `true`, inverts the check, i.e. flags names that are not reserved.
   Default is `false`.

.. option:: AllowedIdentifiers

   Semicolon-separated list of names that the check ignores. Default is an
   empty list.
