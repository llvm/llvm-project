.. title:: clang-tidy - google-explicit-constructor

google-explicit-constructor
===========================


Checks that constructors callable with a single argument and conversion
operators are marked explicit to avoid the risk of unintentional implicit
conversions.

Consider this example:

.. code-block:: c++

  struct S {
    int x;
    operator bool() const { return true; }
  };

  bool f() {
    S a{1};
    S b{2};
    return a == b;
  }

The function will return ``true``, since the objects are implicitly converted to
``bool`` before comparison, which is unlikely to be the intent.

The check will suggest inserting ``explicit`` before the constructor or
conversion operator declaration. However, copy and move constructors should not
be explicit, as well as constructors taking a single ``initializer_list``
argument.

This code:

.. code-block:: c++

  struct S {
    S(int a);
    explicit S(const S&);
    operator bool() const;
    ...

will become

.. code-block:: c++

  struct S {
    explicit S(int a);
    S(const S&);
    explicit operator bool() const;
    ...



See https://google.github.io/styleguide/cppguide.html#Explicit_Constructors
