.. title:: clang-tidy - cert-dcl58-cpp

cert-dcl58-cpp
==============

Modification of the ``std`` or ``posix`` namespace can result in undefined
behavior.
This check warns for such modifications.

Examples:

.. code-block:: c++

  namespace std {
    int x; // May cause undefined behavior.
  }


This check corresponds to the CERT C++ Coding Standard rule
`DCL58-CPP. Do not modify the standard namespaces
<https://www.securecoding.cert.org/confluence/display/cplusplus/DCL58-CPP.+Do+not+modify+the+standard+namespaces>`_.
