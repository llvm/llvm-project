.. title:: clang-tidy - modernize-use-aggregate

modernize-use-aggregate
=======================

Finds classes and structs that could be aggregates if their trivial
forwarding constructors were removed.

A constructor is considered a trivial forwarder when it takes one
parameter per non-static data member, initializes each member from the
corresponding parameter in declaration order, and has an empty body.
Removing such constructors enables aggregate initialization and, in
C++20, designated initializers.

.. code-block:: c++

  // Before
  struct Point {
    int X;
    int Y;
    Point(int X, int Y) : X(X), Y(Y) {}
  };

  Point p(1, 2);

  // After -- remove the constructor
  struct Point {
    int X;
    int Y;
  };

  Point p{1, 2};           // aggregate initialization
  Point q{.X = 1, .Y = 2}; // designated initializers (C++20)

The check will not flag a class if it:

- has virtual functions,
- has private or protected non-static data members,
- has virtual, private, or protected base classes,
- has base classes (before C++17),
- has a user-provided destructor,
- has additional non-trivial constructors, or
- is a template specialization.
