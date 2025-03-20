.. title:: clang-tidy - bugprone-misleading-setter-of-reference

bugprone-misleading-setter-of-reference
=======================================

Finds setter-like member functions that take a pointer parameter and set a
(non-const) reference member of the same class with the pointed value.

The factthat a setter function takes a pointer might cause the belief that an
internal reference (if it would be a pointer) is changed instead of the
pointed-to (or referenced) value.

Only member functions are detected which have a single parameter and contain a
single (maybe overloaded) assignment operator call. The changed member variable
must be private (or protected) for the checker to detect the fault (a public
member can be changed anyway without a set function).

Example:

.. code-block:: c++

  class Widget {
    int& ref_;  // non-const reference member
  public:
    Widget(int &value) : ref_(value) {}

    // Warning: Potentially dangerous setter that could lead to unintended behaviour
    void setRef(int *ptr) {
        ref_ = *ptr;  // This assigns to the referenced value, not changing what ref_ references
    }
  };

  int main() {
    int value1 = 42;
    int value2 = 100;
    Widget w(value1);
    
    // This might look like it changes what ref_ references to,
    // but it actually modifies value1 to be 100
    w.setRef(&value2);  // value1 is now 100, ref_ still references value1
  }

Possible fixes:
  - Change the parameter of the "set" function to non-pointer or const reference
    type.
  - Change the type of the member variable to a pointer and in the "set"
    function assign a value to the pointer (without dereference).
