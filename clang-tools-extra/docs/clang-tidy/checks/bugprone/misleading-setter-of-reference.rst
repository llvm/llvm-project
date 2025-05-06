.. title:: clang-tidy - bugprone-misleading-setter-of-reference

bugprone-misleading-setter-of-reference
=======================================

Finds setter-like member functions that take a pointer parameter and set a
(non-const) reference member of the same class with the pointed value.

The checker detects public member functions that have a single parameter (which
is a pointer) and contain a single (maybe overloaded) assignment operator call.
The assignment should set a member variable with the dereference of the
parameter pointer. The member variable can have any visibility.

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
      InternalRef = *Value;  // This assigns to the referenced value, not changing what ref_ references.
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
  - Change the parameter type of the "set" function to non-pointer or const reference
    type.
  - Change the type of the member variable to a pointer and in the "set"
    function assign a value to the pointer (without dereference).
