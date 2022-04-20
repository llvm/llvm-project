=============================
String Functions in LLVM-libc
=============================

-------
Summary
-------

This site tracks the status of the implementation of string functions in LLVM
Libc. This includes a few extra functions that are not in string.h, such as
functions converting strings to numbers.

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
Function_Name  Available
=============  =========
bzero          YES
bcmp           YES
memcpy         YES
memset         YES
memcmp         YES
memmove        YES
=============  =========


Other Raw Memory Functions
==========================

=============  =========
Function Name  Available
=============  =========
memchr         YES
memrchr        YES
memccpy        YES
mempcpy        YES
=============  =========

String Memory Functions
=======================

=============  =========
Function Name  Available
=============  =========
stpcpy         YES
stpncpy        YES
strcpy         YES
strncpy        YES
strcat         YES
strncat        YES
strdup         YES
strndup        YES
=============  =========

String Examination Functions
============================

=============  =========
Function Name  Available
=============  =========
strlen         YES
strnlen        YES
strcmp         YES
strncmp        YES
strchr         YES
strrchr        YES
strspn         YES
strcspn        YES
strpbrk        YES
strstr         YES
strtok         YES
strtok_r       YES
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
atof           YES
atoi           YES
atol           YES
atoll          YES
strtol         YES
strtoll        YES
strtoul        YES
strtoull       YES
strtof         YES
strtod         YES
strtold        YES
strtoimax      YES
strtoumax      YES
=============  =========

String Error Functions
======================

=============  =========
Function Name  Available
=============  =========
strerror
strerror_s
strerrorlen_s
=============  =========

Localized String Functions
==========================

These functions require locale.h, and will be added when locale support is 
implemented in LLVM-libc.

=============  =========
Function Name  Available
=============  =========
strcoll
strxfrm
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
