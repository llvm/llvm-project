.. title:: clang-tidy - bugprone-public-enable-shared-from-this

bugprone-public-enable-shared-from-this
=======================================

Checks that classes/structs inheriting from ``std::enable_shared_from_this`` derive it with the ``public`` access specifier. If not, then issue a FixItHint that can be applied.

Consider the following code:
.. code-block:: c++
        #include <memory>

        class BadExample : std::enable_shared_from_this<BadExample> {
        // warning: class BadExample is not public even though it's derived from std::enable_shared_from_this [bugprone-public-enable-shared-from-this]
        // will insert the public keyword if -fix is applied 
                public:
                BadExample* foo() { return shared_from_this().get(); }
                void bar() { return; }
        };

        void using_not_public() {
                auto bad_example = std::make_shared<BadExample>();
                auto* b_ex = bad_example->foo();
                b_ex->bar();
        }

When ``using_not_public()`` is called, this code will crash.
