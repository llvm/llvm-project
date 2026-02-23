.. title:: clang-tidy - bugprone-sprintf-to-snprintf

bugprone-sprintf-to-snprintf
============================

Finds calls to ``sprintf`` where the destination is a fixed-size character
array and replaces them with the safer ``snprintf``.

It's a common idiom to have a fixed-size buffer of characters allocated on
the stack and then to ``printf`` into the buffer. This can easily lead to
buffer overflows. This check recommends that the counted version of the
function is used instead.

Example
-------

.. code-block:: c++

  void f() {
    char buff[80];
    sprintf(buff, "Hello, %s!\n", "world");
  }

Becomes:

.. code-block:: c++

  void f() {
    char buff[80];
    snprintf(buff, sizeof(buff), "Hello, %s!\n", "world");
  }