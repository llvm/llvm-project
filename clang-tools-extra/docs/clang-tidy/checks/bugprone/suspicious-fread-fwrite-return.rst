.. title:: clang-tidy - bugprone-suspicious-fread-fwrite-return

bugprone-suspicious-fread-fwrite-return
=======================================

Finds suspicious checks of the return value of ``fread`` and ``fwrite``.

Developers sometimes mistakenly treat the result like the ``ssize_t``
return value of POSIX ``read`` and ``write``. Unlike those functions,
``fread`` and ``fwrite`` return the number of elements transferred as a
``size_t``. When more than one element is requested, comparing the result
against zero does not detect partial reads or writes. Correct code should compare the returned element
count against the requested ``nmemb``.

Examples
--------

.. code-block:: c++

    size_t length = 100;

    // Incorrect: Does not check for short writes.
    if (fwrite(buf, 1, length, fp) <= 0) {
        // ...
    }

    // Incorrect: Discards the short read count.
    if (fread(buf, 1, length, fp) == 0) {
        // ...
    }

    // Incorrect: Tautological condition.
    size_t written = fwrite(buf, 1, length, fp);
    if (written < 0) {
        // ...
    }

    // Correct: Compare against the requested number of items.
    if (fwrite(buf, 1, length, fp) != length) {
        // ...
    }
