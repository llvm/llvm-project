.. title:: clang-tidy - readability-named-parameter

readability-named-parameter
===========================

Find functions with unnamed arguments.

The check implements the following rule originating in the Google C++ Style
Guide:

https://google.github.io/styleguide/cppguide.html#Function_Declarations_and_Definitions

A parameter name may be omitted only if the parameter is not used in the
function's definition.

.. code-block:: c++
int doingSomething(int a, int b, int) {  // Ok: the third paramet is not used
    return a + b;
}

Corresponding cpplint.py check name: `readability/function`.
