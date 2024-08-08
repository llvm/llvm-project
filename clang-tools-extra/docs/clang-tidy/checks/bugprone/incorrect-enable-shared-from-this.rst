.. title:: clang-tidy - bugprone-incorrect-enable-shared-from-this

bugprone-incorrect-enable-shared-from-this
=======================================

Check if class/struct publicly derives from ``std::enable_shared_from_this``,
because otherwise when ``shared_from_this`` is called it will throw 
``std::bad_weak_ptr``. Issues a ``FixItHint`` that can be applied.

Consider the following code:

.. code-block:: c++
    #include <memory>

    class BadExample : std::enable_shared_from_this<BadExample> {
    // warning: inheritance from std::enable_shared_from_this 
    // should be public inheritance,
    // otherwise the internal weak_ptr won't be initialized 
    // [bugprone-incorrect-enable-shared-from-this]
        public:
        BadExample* foo() { return shared_from_this().get(); }
        void bar() { return; }
    };

    void using_not_public() {
        auto bad_example = std::make_shared<BadExample>();
        auto* b_ex = bad_example->foo();
        b_ex->bar();
    }

When ``using_not_public()`` is called, this code will crash without exception 
handling.
