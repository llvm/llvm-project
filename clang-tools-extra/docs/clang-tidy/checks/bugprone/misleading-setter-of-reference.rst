.. title:: clang-tidy - bugprone-misleading-setter-of-reference

bugprone-misleading-setter-of-reference
=======================================

Finds setter-like member functions that take a pointer parameter and set a
reference member of the same class with the pointed value.

The check detects member functions that take a single pointer parameter,
and contain a single expression statement that dereferences the parameter and
assigns the result to a data member with a reference type.

The fact that a setter function takes a pointer might cause the belief that an
internal reference (if it would be a pointer) is changed instead of the
pointed-to (or referenced) value.

Example:

.. code-block:: c++

  class MyClass {
    int &InternalRef;  // non-const reference member
  public:
    MyClass(int &Value) : InternalRef(Value) {}

    // Warning: This setter could lead to unintended behaviour.
    void setRef(int *Value) {
      InternalRef = *Value;  // This assigns to the referenced value, not changing what InternalRef references.
    }
  };

  int main() {
    int Value1 = 42;
    int Value2 = 100;
    MyClass X(Value1);

    // This might look like it changes what InternalRef references to,
    // but it actually modifies Value1 to be 100.
    X.setRef(&Value2);
  }

Possible fixes:
  - Change the parameter type of the "set" function to non-pointer type (for
    example, a const reference).
  - Change the type of the member variable to a pointer and in the "set"
    function assign a value to the pointer (without dereference).
