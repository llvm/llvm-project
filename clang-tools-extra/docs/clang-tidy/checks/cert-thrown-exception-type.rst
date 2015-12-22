.. title:: clang-tidy - cert-err60-cpp

cert-err60-cpp
==============

This check flags all throw expressions where the exception object is not nothrow
copy constructible.

This check corresponds to the CERT C++ Coding Standard rule
`ERR60-CPP. Exception objects must be nothrow copy constructible
<https://www.securecoding.cert.org/confluence/display/cplusplus/ERR60-CPP.+Exception+objects+must+be+nothrow+copy+constructible>`_.
