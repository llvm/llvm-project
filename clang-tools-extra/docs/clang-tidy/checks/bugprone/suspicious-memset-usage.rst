.. title:: clang-tidy - bugprone-suspicious-memset-usage

bugprone-suspicious-memset-usage
================================

This check finds ``memset()`` calls with potential mistakes in their arguments.
Considering the function as ``void* memset(void* destination, int fill_value,
size_t byte_count)``, the following cases are covered:

**Case 1: Fill value is a character ``'0'``**

Filling up a memory area with ASCII code 48 characters is not customary,
possibly integer zeroes were intended instead.
The check offers a replacement of ``'0'`` with ``0``. Memsetting character
pointers with ``'0'`` is allowed.

**Case 2: Fill value is truncated**

Memset converts ``fill_value`` to ``unsigned char`` before using it. If
``fill_value`` is out of unsigned character range, it gets truncated
and memory will not contain the desired pattern.

**Case 3: Byte count is zero**

Calling memset with a literal zero in its ``byte_count`` argument is likely
to be unintended and swapped with ``fill_value``. The check offers to swap
these two arguments.

Corresponding cpplint.py check name: ``runtime/memset``.


Examples:

.. code-block:: c++

  void foo() {
    int i[5] = {1, 2, 3, 4, 5};
    int *ip = i;
    char c = '1';
    char *cp = &c;
    int v = 0;

    // Case 1
    memset(ip, '0', 1); // suspicious
    memset(cp, '0', 1); // OK

    // Case 2
    memset(ip, 0xabcd, 1); // fill value gets truncated
    memset(ip, 0x00, 1);   // OK

    // Case 3
    memset(ip, sizeof(int), v); // zero length, potentially swapped
    memset(ip, 0, 1);           // OK
  }
