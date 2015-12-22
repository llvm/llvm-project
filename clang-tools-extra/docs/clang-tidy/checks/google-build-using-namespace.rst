.. title:: clang-tidy - google-build-using-namespace

google-build-using-namespace
============================


Finds using namespace directives.

http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml?showone=Namespaces#Namespaces

The check implements the following rule of the Google C++ Style Guide:

  You may not use a using-directive to make all names from a namespace
  available.

  .. code:: c++

    // Forbidden -- This pollutes the namespace.
    using namespace foo;

Corresponding cpplint.py check name: ``build/namespaces``.
