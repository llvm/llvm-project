================
String Functions
================

.. include:: check.rst

---------------
Source location
---------------

-   The main source for string functions is located at:
    ``libc/src/string``.

-   The source for string conversion functions is located at:
    ``libc/src/stdlib`` and
    ``libc/src/__support``.

-   The tests are located at:
    ``libc/test/src/string``,
    ``libc/test/src/stdlib``, and
    ``libc/test/src/__support``
    respectively.

---------------------
Implementation Status
---------------------

Primary memory functions
========================

.. TODO(gchatelet): add details about the memory functions.


=============  =========
Function Name  Available
=============  =========
bzero          |check|
bcmp           |check|
bcopy          |check|       
memcpy         |check|
memset         |check|
memcmp         |check|
memmove        |check|
=============  =========


Other Raw Memory Functions
==========================

=============  =========
Function Name  Available
=============  =========
memchr         |check|
memrchr        |check|
memccpy        |check|
mempcpy        |check|
=============  =========

String Memory Functions
=======================

=============  =========
Function Name  Available
=============  =========
stpcpy         |check|
stpncpy        |check|
strcpy         |check|
strncpy        |check|
strcat         |check|
strncat        |check|
strdup         |check|
strndup        |check|
=============  =========

String Examination Functions
============================

=============  =========
Function Name  Available
=============  =========
strlen         |check|
strnlen        |check|
strcmp         |check|
strncmp        |check|
strchr         |check|
strrchr        |check|
strspn         |check|
strcspn        |check|
strpbrk        |check|
strstr         |check|
strtok         |check|
strtok_r       |check|
=============  =========

String Conversion Functions
============================

These functions are not in strings.h, but are still primarily string
functions, and are therefore tracked along with the rest of the string
functions.

The String to float functions were implemented using the Eisel-Lemire algorithm 
(read more about the algorithm here: `The Eisel-Lemire ParseNumberF64 Algorithm
<https://nigeltao.github.io/blog/2020/eisel-lemire.html>`_). This improved
the performance of string to float and double, and allowed it to complete this
comprehensive test 15% faster than glibc: `Parse Number FXX Test Data
<https://github.com/nigeltao/parse-number-fxx-test-data>`_. The test was done 
with LLVM-libc built on 2022-04-14 and Debian GLibc version 2.33-6. The targets
``libc_str_to_float_comparison_test`` and 
``libc_system_str_to_float_comparison_test`` were built and run on the test data
10 times each, skipping the first run since it was an outlier.


=============  =========
Function Name  Available
=============  =========
atof           |check|
atoi           |check|
atol           |check|
atoll          |check|
strtol         |check|
strtoll        |check|
strtoul        |check|
strtoull       |check|
strtof         |check|
strtod         |check|
strtold        |check|
strtoimax      |check|
strtoumax      |check|
=============  =========

String Error Functions
======================

=============  =========
Function Name  Available
=============  =========
strerror       |check|
strerror_r     |check|
=============  =========

Localized String Functions
==========================

These functions require locale.h, and will be finished when locale support is 
implemented in LLVM-libc.

=============  =========
Function Name  Available
=============  =========
strcoll        Partially
strxfrm        Partially
=============  =========

---------------------------
\<name\>_s String Functions
---------------------------

Many String functions have an equivalent _s version, which is intended to be
more secure and safe than the previous standard. These functions add runtime
error detection and overflow protection. While they can be seen as an
improvement, adoption remains relatively low among users. In addition, they are
being considered for removal, see 
`Field Experience With Annex K — Bounds Checking Interfaces
<http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1967.htm>`_. For these reasons, 
there is no ongoing work to implement them.
