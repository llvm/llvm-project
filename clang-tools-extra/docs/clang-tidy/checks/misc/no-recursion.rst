.. title:: clang-tidy - misc-no-recursion

misc-no-recursion
=================

Finds strongly connected functions (by analyzing the call graph for
SCC's (Strongly Connected Components) that are loops),
diagnoses each function in the cycle,
and displays one example of a possible call graph loop (recursion).

References:

* CERT C++ Coding Standard rule `DCL56-CPP. Avoid cycles during initialization of static objects <https://wiki.sei.cmu.edu/confluence/display/cplusplus/DCL56-CPP.+Avoid+cycles+during+initialization+of+static+objects>`_.
* JPL Institutional Coding Standard for the C Programming Language (JPL DOCID D-60411) rule `2.4 Do not use direct or indirect recursion`.
* OpenCL Specification, Version 1.2 rule `6.9 Restrictions: i. Recursion is not supported. <https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf>`_.

Limitations:

* The check does not handle calls done through function pointers
* The check does not handle C++ destructors
