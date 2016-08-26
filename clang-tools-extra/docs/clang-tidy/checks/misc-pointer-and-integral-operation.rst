.. title:: clang-tidy - misc-pointer-and-integral-operation

misc-pointer-and-integral-operation
===================================

Looks for operation involving pointers and integral types. A common mistake is
to forget to dereference a pointer. These errors may be detected when a pointer
object is compare to an object with integral type.

Examples:

.. code-block:: c++

  char* ptr;
  if ((ptr = malloc(...)) < nullptr)   // Pointer comparison with operator '<'
    ...                                // Should probably be '!='

  if (ptr == '\0')   // Should probably be *ptr
    ... 

  void Process(std::string path, bool* error) {
    [...]
    if (error != false)  // Should probably be *error
      ...
