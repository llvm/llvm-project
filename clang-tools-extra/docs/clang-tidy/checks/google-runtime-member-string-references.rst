.. title:: clang-tidy - google-runtime-member-string-references

google-runtime-member-string-references
=======================================

Finds members of type ``const string&``.

const string reference members are generally considered unsafe as they can be
created from a temporary quite easily.

.. code-block:: c++

  struct S {
    S(const string &Str) : Str(Str) {}
    const string &Str;
  };
  S instance("string");

In the constructor call a string temporary is created from ``const char *`` and
destroyed immediately after the call. This leaves around a dangling reference.

This check emit warnings for both ``std::string`` and ``::string`` const
reference members.

Corresponding cpplint.py check name: `runtime/member_string_reference`.
