.. title:: clang-tidy - bugprone-copy-constructor-init

bugprone-copy-constructor-init
==============================

Finds copy constructors where the constructor doesn't call the copy constructor
of the base class.

.. code-block:: c++

    class Copyable {
    public:
      Copyable() = default;
      Copyable(const Copyable &) = default;

      int memberToBeCopied = 0;
    };

    class X2 : public Copyable {
      X2(const X2 &other) {} // Copyable(other) is missing
    };

Also finds copy constructors where the constructor of
the base class don't have parameter.

.. code-block:: c++

    class X3 : public Copyable {
      X3(const X3 &other) : Copyable() {} // other is missing
    };

Failure to properly initialize base class sub-objects during copy construction
can result in undefined behavior, crashes, data corruption, or other unexpected
outcomes. The check ensures that the copy constructor of a derived class
properly calls the copy constructor of the base class, helping to prevent bugs
and improve code quality.

Limitations:

* It won't generate warnings for empty classes, as there are no class members
  (including base class sub-objects) to worry about.

* It won't generate warnings for base classes that have copy constructor
  private or deleted.

* It won't generate warnings for base classes that are initialized using other
  non-default constructor, as this could be intentional.

The check also suggests a fix-its in some cases.
