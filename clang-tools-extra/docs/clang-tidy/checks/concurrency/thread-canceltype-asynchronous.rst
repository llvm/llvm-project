.. title:: clang-tidy - concurrency-thread-canceltype-asynchronous

concurrency-thread-canceltype-asynchronous
==========================================

Finds ``pthread_setcanceltype`` function calls where a thread's cancellation
type is set to asynchronous. Asynchronous cancellation type
(``PTHREAD_CANCEL_ASYNCHRONOUS``) is generally unsafe, use type
``PTHREAD_CANCEL_DEFERRED`` instead which is the default. Even with deferred
cancellation, a cancellation point in an asynchronous signal handler may still
be acted upon and the effect is as if it was an asynchronous cancellation.

.. code-block:: c++

  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &oldtype);

This check corresponds to the CERT C Coding Standard rule
`POS47-C. Do not use threads that can be canceled asynchronously
<https://wiki.sei.cmu.edu/confluence/display/c/POS47-C.+Do+not+use+threads+that+can+be+canceled+asynchronously>`_.

`cert-pos47-c` redirects here as an alias of this check.
