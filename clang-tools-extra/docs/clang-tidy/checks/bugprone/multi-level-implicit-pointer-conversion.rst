.. title:: clang-tidy - bugprone-multi-level-implicit-pointer-conversion

bugprone-multi-level-implicit-pointer-conversion
================================================

Detects implicit conversions between pointers of different levels of
indirection.

Conversions between pointer types of different levels of indirection can be
dangerous and may lead to undefined behavior, particularly if the converted
pointer is later cast to a type with a different level of indirection.
For example, converting a pointer to a pointer to an ``int`` (``int**``) to
a ``void*`` can result in the loss of information about the original level of
indirection, which can cause problems when attempting to use the converted
pointer. If the converted pointer is later cast to a type with a different
level of indirection and dereferenced, it may lead to access violations,
memory corruption, or other undefined behavior.

Consider the following example:

.. code-block:: c++

  void foo(void* ptr);

  int main() {
    int x = 42;
    int* ptr = &x;
    int** ptr_ptr = &ptr;
    foo(ptr_ptr); // warning will trigger here
    return 0;
  }

In this example, ``foo()`` is called with ``ptr_ptr`` as its argument. However,
``ptr_ptr`` is a ``int**`` pointer, while ``foo()`` expects a ``void*`` pointer.
This results in an implicit pointer level conversion, which could cause issues
if ``foo()`` dereferences the pointer assuming it's a ``int*`` pointer.

Using an explicit cast is a recommended solution to prevent issues caused by
implicit pointer level conversion, as it allows the developer to explicitly
state their intention and show their reasoning for the type conversion.
Additionally, it is recommended that developers thoroughly check and verify the
safety of the conversion before using an explicit cast. This extra level of
caution can help catch potential issues early on in the development process,
improving the overall reliability and maintainability of the code.
