.. title:: clang-tidy - bugprone-move-forwarding-reference

bugprone-move-forwarding-reference
==================================

Warns if ``std::move`` is called on a forwarding reference, for example:

  .. code-block:: c++

    template <typename T>
    void foo(T&& t) {
      bar(std::move(t));
    }

`Forwarding references
<http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4164.pdf>`_ should
typically be passed to ``std::forward`` instead of ``std::move``, and this is
the fix that will be suggested.

(A forwarding reference is an rvalue reference of a type that is a deduced
function template argument.)

In this example, the suggested fix would be

  .. code-block:: c++

    bar(std::forward<T>(t));

Background
----------

Code like the example above is sometimes written with the expectation that
``T&&`` will always end up being an rvalue reference, no matter what type is
deduced for ``T``, and that it is therefore not possible to pass an lvalue to
``foo()``. However, this is not true. Consider this example:

  .. code-block:: c++

    std::string s = "Hello, world";
    foo(s);

This code compiles and, after the call to ``foo()``, ``s`` is left in an
indeterminate state because it has been moved from. This may be surprising to
the caller of ``foo()`` because no ``std::move`` was used when calling
``foo()``.

The reason for this behavior lies in the special rule for template argument
deduction on function templates like ``foo()`` -- i.e. on function templates
that take an rvalue reference argument of a type that is a deduced function
template argument. (See section [temp.deduct.call]/3 in the C++11 standard.)

If ``foo()`` is called on an lvalue (as in the example above), then ``T`` is
deduced to be an lvalue reference. In the example, ``T`` is deduced to be
``std::string &``. The type of the argument ``t`` therefore becomes
``std::string& &&``; by the reference collapsing rules, this collapses to
``std::string&``.

This means that the ``foo(s)`` call passes ``s`` as an lvalue reference, and
``foo()`` ends up moving ``s`` and thereby placing it into an indeterminate
state.
