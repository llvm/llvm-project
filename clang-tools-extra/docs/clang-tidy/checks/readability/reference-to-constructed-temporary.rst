.. title:: clang-tidy - readability-reference-to-constructed-temporary

readability-reference-to-constructed-temporary
==============================================

Detects C++ code where a reference variable is used to extend the lifetime of
a temporary object that has just been constructed.

This construction is often the result of multiple code refactorings or a lack
of developer knowledge, leading to confusion or subtle bugs. In most cases,
what the developer really wanted to do is create a new variable rather than
extending the lifetime of a temporary object.

Examples of problematic code include:

.. code-block:: c++

   const std::string& str("hello");

   struct Point { int x; int y; };
   const Point& p = { 1, 2 };

In the first example, a ``const std::string&`` reference variable ``str`` is
assigned a temporary object created by the ``std::string("hello")``
constructor. In the second example, a ``const Point&`` reference variable ``p``
is assigned an object that is constructed from an initializer list ``{ 1, 2 }``.
Both of these examples extend the lifetime of the temporary object to the
lifetime of the reference variable, which can make it difficult to reason about
and may lead to subtle bugs or misunderstanding.

To avoid these issues, it is recommended to change the reference variable to a
(``const``) value variable.
