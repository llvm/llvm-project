.. title:: clang-tidy - misc-anonymous-namespace-in-header

misc-anonymous-namespace-in-header
==================================

Finds anonymous namespaces in headers.

Anonymous namespaces in headers can lead to One Definition Rule (ODR)
violations because each translation unit including the header will get its
own unique version of the symbols. This increases binary size and can cause
confusing link-time errors.

References
----------

This check corresponds to the CERT C++ Coding Standard rule
`DCL59-CPP. Do not define an unnamed namespace in a header file
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/DCL59-CPP.+Do+not+define+an+unnamed+namespace+in+a+header+file>`_.

Corresponding cpplint.py check name: `build/namespaces`.
