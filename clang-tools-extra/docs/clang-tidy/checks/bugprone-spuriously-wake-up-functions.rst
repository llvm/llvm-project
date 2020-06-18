.. title:: clang-tidy - bugprone-spuriously-wake-up-functions

bugprone-spuriously-wake-up-functions
=====================================

Finds ``cnd_wait``, ``cnd_timedwait``, ``wait``, ``wait_for``, or 
``wait_until`` function calls when the function is not invoked from a loop
that checks whether a condition predicate holds or the function has a 
condition parameter.

.. code-block: c++

    if (condition_predicate) {
        condition.wait(lk);
    }

.. code-block: c

    if (condition_predicate) {
        if (thrd_success != cnd_wait(&condition, &lock)) {
        }
    }

This check corresponds to the CERT C++ Coding Standard rule
`CON54-CPP. Wrap functions that can spuriously wake up in a loop
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/CON54-CPP.+Wrap+functions+that+can+spuriously+wake+up+in+a+loop>`_.
and CERT C Coding Standard rule
`CON36-C. Wrap functions that can spuriously wake up in a loop
<https://wiki.sei.cmu.edu/confluence/display/c/CON36-C.+Wrap+functions+that+can+spuriously+wake+up+in+a+loop>`_.
