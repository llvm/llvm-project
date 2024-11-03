.. title:: clang-tidy - cppcoreguidelines-avoid-capturing-lambda-coroutines

cppcoreguidelines-avoid-capturing-lambda-coroutines
===================================================

Flags C++20 coroutine lambdas with non-empty capture lists that may cause
use-after-free errors and suggests avoiding captures or ensuring the lambda
closure object has a guaranteed lifetime.

This check implements `CP.51
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rcoro-capture>`_
from the C++ Core Guidelines.

Using coroutine lambdas with non-empty capture lists can be risky, as capturing
variables can lead to accessing freed memory after the first suspension point.
This issue can occur even with refcounted smart pointers and copyable types.
When a lambda expression creates a coroutine, it results in a closure object
with storage, which is often on the stack and will eventually go out of scope.
When the closure object goes out of scope, its captures also go out of scope.
While normal lambdas finish executing before this happens, coroutine lambdas may
resume from suspension after the closure object has been destructed, resulting
in use-after-free memory access for all captures.

Consider the following example:

.. code-block:: c++

    int value = get_value();
    std::shared_ptr<Foo> sharedFoo = get_foo();
    {
        const auto lambda = [value, sharedFoo]() -> std::future<void>
        {
            co_await something();
            // "sharedFoo" and "value" have already been destroyed
            // the "shared" pointer didn't accomplish anything
        };
        lambda();
    } // the lambda closure object has now gone out of scope

In this example, the lambda object is defined with two captures: value and
``sharedFoo``. When ``lambda()`` is called, the lambda object is created on the
stack, and the captures are copied into the closure object. When the coroutine
is suspended, the lambda object goes out of scope, and the closure object is
destroyed. When the coroutine is resumed, the captured variables may have been
destroyed, resulting in use-after-free bugs.

In conclusion, the use of coroutine lambdas with non-empty capture lists can
lead to use-after-free errors when resuming the coroutine after the closure
object has been destroyed. This check helps prevent such errors by flagging
C++20 coroutine lambdas with non-empty capture lists and suggesting avoiding
captures or ensuring the lambda closure object has a guaranteed lifetime.

Following these guidelines can help ensure the safe and reliable use of
coroutine lambdas in C++ code.
