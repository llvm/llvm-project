.. title:: clang-tidy - bugprone-throwing-static-initialization

bugprone-throwing-static-initialization
=======================================

Finds all ``static`` or ``thread_local`` variable declarations where the
initializer for the object may throw an exception.

Options
-------

.. option:: AllowedTypes

  A semicolon-separated list of names of types that will be excluded from
  this check (declarations with matching type will be excluded). Regular
  expressions are accepted, e.g. ``[Rr]ef(erence)?$`` matches every type with
  suffix ``Ref``, ``ref``, ``Reference`` and ``reference``. If a name in the
  list contains the sequence `::`, it is matched against the qualified type
  name (i.e. ``namespace::Type``), otherwise it is matched against only the
  type name (i.e. ``Type``). Default is an empty string.

References
----------

This check corresponds to the CERT C++ Coding Standard rule
`ERR58-CPP. Handle all exceptions thrown before main() begins executing
<https://www.securecoding.cert.org/confluence/display/cplusplus/ERR58-CPP.+Handle+all+exceptions+thrown+before+main%28%29+begins+executing>`_.
