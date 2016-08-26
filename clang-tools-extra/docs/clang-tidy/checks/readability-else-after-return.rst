.. title:: clang-tidy - readability-else-after-return

readability-else-after-return
=============================

`LLVM Coding Standards <http://llvm.org/docs/CodingStandards.html>`_ advises to
reduce indentation where possible and where it makes understanding code easier.
Early exit is one of the suggested enforcements of that. Please do not use
``else`` or ``else if`` after something that interrupts control flow - like
``return``, ``break``, ``continue``, ``throw``.

The following piece of code illustrates how the check works. This piece of code:

.. code-block:: c++

    void foo(int Value) {
      int Local = 0;
      for (int i = 0; i < 42; i++) {
        if (Value == 1) {
          return;
        } else {
          Local++;
        }

        if (Value == 2)
          continue;
        else
          Local++;

        if (Value == 3) {
          throw 42;
        } else {
          Local++;
        }
      }
    }


Would be transformed into:

.. code-block:: c++

    void foo(int Value) {
      int Local = 0;
      for (int i = 0; i < 42; i++) {
        if (Value == 1) {
          return;
        }
        Local++;

        if (Value == 2)
          continue;
        Local++;

        if (Value == 3) {
          throw 42;
        }
        Local++;
      }
    }


This check helps to enforce this `LLVM Coding Standards recommendation
<http://llvm.org/docs/CodingStandards.html#don-t-use-else-after-a-return>`_.
