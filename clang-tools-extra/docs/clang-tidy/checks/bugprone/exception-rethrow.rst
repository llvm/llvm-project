.. title:: clang-tidy - bugprone-exception-rethrow

bugprone-exception-rethrow
==========================

Identifies problematic exception rethrowing, especially with caught exception
variables or empty throw statements outside catch blocks.

In C++ exception handling, a common pitfall occurs when developers rethrow
caught exceptions within catch blocks by directly passing the caught exception
variable to the ``throw`` statement. While this approach can propagate
exceptions to higher levels of the program, it often leads to code that is less
clear and more error-prone. Rethrowing caught exceptions with the same exception
object within catch blocks can obscure the original context of the exception and
make it challenging to trace program flow. Additionally, this method can
introduce issues such as exception object slicing and performance overhead due
to the invocation of the copy constructor.

.. code-block:: c++

  try {
    // Code that may throw an exception
  } catch (const std::exception& e) {
    throw e; // Bad
  }

To prevent these issues, it is advisable to utilize ``throw;`` statements to
rethrow the original exception object for currently handled exceptions.

.. code-block:: c++

  try {
    // Code that may throw an exception
  } catch (const std::exception& e) {
    throw; // Good
  }

However, when empty throw statement is used outside of a catch block, it
will result in a call to ``std::terminate()``, which abruptly terminates the
application. This behavior can lead to abnormal termination of the program and
is often unintended. Such occurrences may indicate errors or oversights in the
exception handling logic, and it is essential to avoid empty throw statements
outside catch blocks to prevent unintended program termination.

.. code-block:: c++

  void foo() {
    // std::terminate will be called because there is no exception to rethrow
    throw;
  }

  int main() {
    try {
      foo();
    } catch(...) {
      return 1;
    }
    return 0;
  }

Above program will be terminated with:

.. code-block:: text

  terminate called without an active exception
  Aborted (core dumped)


