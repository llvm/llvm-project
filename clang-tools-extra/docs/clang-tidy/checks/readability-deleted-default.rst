.. title:: clang-tidy - readability-deleted-default

readability-deleted-default
===========================

Checks that constructors and assignment operators marked as ``= default`` are
not actually deleted by the compiler.

.. code-block:: c++

  class Example {
  public:
    // This constructor is deleted because I is missing a default value.
    Example() = default;
    // This is fine.
    Example(const Example& Other) = default;
    // This operator is deleted because I cannot be assigned (it is const).
    Example& operator=(const Example& Other) = default;

  private:
    const int I;
  };
