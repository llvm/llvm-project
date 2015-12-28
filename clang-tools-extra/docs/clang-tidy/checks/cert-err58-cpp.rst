.. title:: clang-tidy - cert-err58-cpp

cert-err58-cpp
==============

This check flags all static or thread_local variable declarations where the
constructor for the object may throw an exception.

This check corresponds to the CERT C++ Coding Standard rule
`ERR58-CPP. Constructors of objects with static or thread storage duration must not throw exceptions
<https://www.securecoding.cert.org/confluence/display/cplusplus/ERR58-CPP.+Constructors+of+objects+with+static+or+thread+storage+duration+must+not+throw+exceptions>`_.
