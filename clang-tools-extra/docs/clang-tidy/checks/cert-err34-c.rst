.. title:: clang-tidy - cert-err34-c

cert-err34-c
============

This check flags calls to string-to-number conversion functions that do not
verify the validity of the conversion, such as ``atoi()`` or ``scanf()``. It
does not flag calls to ``strtol()``, or other, related conversion functions that
do perform better error checking.

.. code-block:: c

  #include <stdlib.h>
  
  void func(const char *buff) {
    int si;
    
    if (buff) {
      si = atoi(buff); /* 'atoi' used to convert a string to an integer, but function will
                           not report conversion errors; consider using 'strtol' instead. */
    } else {
      /* Handle error */
    }
  }

This check corresponds to the CERT C Coding Standard rule
`ERR34-C. Detect errors when converting a string to a number
<https://www.securecoding.cert.org/confluence/display/c/ERR34-C.+Detect+errors+when+converting+a+string+to+a+number>`_.
