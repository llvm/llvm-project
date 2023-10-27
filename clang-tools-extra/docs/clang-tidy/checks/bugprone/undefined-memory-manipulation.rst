.. title:: clang-tidy - bugprone-undefined-memory-manipulation

bugprone-undefined-memory-manipulation
======================================

Finds calls of memory manipulation functions ``memset()``, ``memcpy()`` and
``memmove()`` on non-TriviallyCopyable objects resulting in undefined behavior.

Using memory manipulation functions on non-TriviallyCopyable objects can lead
to a range of subtle and challenging issues in C++ code. The most immediate
concern is the potential for undefined behavior, where the state of the object
may become corrupted or invalid. This can manifest as crashes, data corruption,
or unexpected behavior at runtime, making it challenging to identify and
diagnose the root cause. Additionally, misuse of memory manipulation functions
can bypass essential object-specific operations, such as constructors and
destructors, leading to resource leaks or improper initialization.

For example, when using ``memcpy`` to copy ``std::string``, pointer data is
being copied, and it can result in a double free issue.

.. code-block:: c++

  #include <cstring>
  #include <string>

  int main() {
      std::string source = "Hello";
      std::string destination;

      std::memcpy(&destination, &source, sizeof(std::string));

      // Undefined behavior may occur here, during std::string destructor call.
      return 0;
  }
