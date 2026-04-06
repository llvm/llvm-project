.. title:: clang-tidy - readability-container-data-pointer

readability-container-data-pointer
==================================

Finds cases where code could use ``data()`` or ``c_str()`` rather than the
address of the element at index 0 in a container. This pattern is commonly used
to materialize a pointer to the backing data of a container. ``std::vector`` and
``std::string`` provide a ``data()`` accessor to retrieve the data pointer
which should be preferred.

This check suggests ``data()`` for non-const pointer contexts and ``c_str()``
for const pointer contexts when available. This provides better semantic
clarity: ``c_str()`` explicitly indicates read-only access to string data,
while ``data()`` may allow modifications.

This also ensures that in the case that the container is empty, the data
pointer access does not perform an errant memory access.

Examples
--------

.. code-block: c++

  std::string s;
  std::vector<int> v;

  char* p1 = &s[0];           // Warning: use s.data()
  const char* p2 = &s[0]      // Warning: use s.c_str()
  int* p3 = &v[0];            // Warning: use v.data()

  const std::string cs;
  const char* p4 = &cs[0];    // Warning: use cs.c_str()

Options
-------

.. option:: IgnoredContainers

   Semicolon-separated list of containers regexp for which this check won't be
   enforced. Default is an empty string.
