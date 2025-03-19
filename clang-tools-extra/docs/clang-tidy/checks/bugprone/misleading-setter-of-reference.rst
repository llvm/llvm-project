.. title:: clang-tidy - bugprone-misleading-setter-of-reference

bugprone-misleading-setter-of-reference
=======================================

Finds setter-like functions that take a pointer parameter and set a (non-const)
reference with the pointed value. The fact that a setter function takes a
pointer might cause the belief that an internal reference (if it would be a
pointer) is changed instead of the pointed-to (or referenced) value.

Only member functions are detected which have a single parameter and contain a
single (maybe overloaded) assignment operator call.

Example:

.. code-block:: c++

  class Widget {
    int& ref_;  // non-const reference member
  public:
    // non-copyable
    Widget(const Widget&) = delete;
    // non-movable
    Widget(Widget&&) = delete;
 
    Widget(int& value) : ref_(value) {}
    
    // Potentially dangerous setter that could lead to unintended behaviour
    void setRef(int* ptr) {
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
