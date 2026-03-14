.. title:: clang-tidy - performance-avoid-endl

performance-avoid-endl
============================

Checks for uses of ``std::endl`` on streams and suggests using the newline
character ``'\n'`` instead.

Rationale:
Using ``std::endl`` on streams can be less efficient than using the newline
character ``'\n'`` because ``std::endl`` performs two operations: it writes a
newline character to the output stream and then flushes the stream buffer.
Writing a single newline character using ``'\n'`` does not trigger a flush,
which can improve performance. In addition, flushing the stream buffer can
cause additional overhead when working with streams that are buffered.

Example:

Consider the following code:

.. code-block:: c++

    #include <iostream>

    int main() {
      std::cout << "Hello" << std::endl;
    }

Which gets transformed into:

.. code-block:: c++

    #include <iostream>

    int main() {
      std::cout << "Hello" << '\n';
    }

This code writes a single newline character to the ``std::cout`` stream without
flushing the stream buffer.

Additionally, it is important to note that the standard C++ streams (like
``std::cerr``, ``std::wcerr``, ``std::clog`` and ``std::wclog``)
always flush after a write operation, unless ``std::ios_base::sync_with_stdio``
is set to ``false``. regardless of whether ``std::endl`` or ``'\n'`` is used.
Therefore, using ``'\n'`` with these streams will not
result in any performance gain, but it is still recommended to use
``'\n'`` for consistency and readability.

If you do need to flush the stream buffer, you can use ``std::flush``
explicitly like this:

.. code-block:: c++

    #include <iostream>

    int main() {
      std::cout << "Hello\n" << std::flush;
    }
