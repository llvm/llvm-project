.. title:: clang-tidy - bugprone-eval-order

bugprone-eval-order
===================

Order of evaluation is unspecified for function and constructor parameters,
unless the constructor uses C++11 list-initialization.

This may lead to issues in code that is written with the assumption of a
specified evaluation order.

This tidy rule will print a warning when at least two parameters in a call are
sourced from any method of a class, and at least one of those method calls is
non-``const``. A fix is not offered.

C++11 constructor list-initialization are evaluated in order of appearance and
can be used as a fix.

In case of false positives, the check can be suppressed, or disabled completely.

``static`` methods and global functions may also lead to issues, but they are
not considered in this check, for now.

The check is limited to member functions, because presumably the most common
issue in real code is some kind of reader object:

.. code-block:: c++

  struct Reader {
      char buf[4]{};
      int pos{};
      char Pop() { return buf[pos++]; }
  };

  int Calc(char byte1, char byte2) { return byte1 > byte2; };

  int main() {
      Reader r{{1, 2, 3}};
      return Calc(r.Pop(), r.Pop());  // May return 0 or 1
  }
