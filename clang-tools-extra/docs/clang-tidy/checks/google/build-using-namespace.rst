.. title:: clang-tidy - google-build-using-namespace

google-build-using-namespace
============================

Finds ``using namespace`` directives.

The check implements the following rule of the
`Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html#Namespaces>`_:

  You may not use a using-directive to make all names from a namespace
  available.

.. code-block:: c++

    // Forbidden -- This pollutes the namespace.
    using namespace foo;

Corresponding cpplint.py check name: `build/namespaces`.
