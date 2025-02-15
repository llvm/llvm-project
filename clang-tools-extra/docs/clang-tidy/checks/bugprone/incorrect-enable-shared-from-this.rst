.. title:: clang-tidy - bugprone-incorrect-enable-shared-from-this

bugprone-incorrect-enable-shared-from-this
==========================================

Detect classes or structs that do not publicly inherit from 
``std::enable_shared_from_this``, because unintended behavior will 
otherwise occur when calling ``shared_from_this``.

Consider the following code:

.. code-block:: c++

    #include <memory>

    // private inheritance
    class BadExample : std::enable_shared_from_this<BadExample> {
    
    // ``shared_from_this``` unintended behaviour
    // `libstdc++` implementation returns uninitialized ``weak_ptr``
        public:
        BadExample* foo() { return shared_from_this().get(); }
        void bar() { return; }
    };

    void using_not_public() {
        auto bad_example = std::make_shared<BadExample>();
        auto* b_ex = bad_example->foo();
        b_ex->bar();
    }

Using `libstdc++` implementation, ``shared_from_this`` will throw 
``std::bad_weak_ptr``. When ``using_not_public()`` is called, this code will 
crash without exception handling.
