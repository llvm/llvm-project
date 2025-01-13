.. title:: clang-tidy - performance-explicit-move-constructor

performance-explicit-move-constructor
=====================================

Warns when a class defines an explicit move constructor, which may cause
the copy constructor to get called instead.

Example:

.. code-block:: c++

    class Expensive {
    public:
        // ...
        Expensive(const Expensive&) { /* ... */ }
        explicit Expensive(Expensive&&) { /* ... */ }
    };

    void process(Expensive);

    int main() {
        Expensive exp{};
        process(std::move(exp));

        return 0;
    }

Here, the call to ``process`` is actually going to copy ``exp`` instead of
moving it, potentially incurring a performance penalty if copying is expensive.
No warning will be emitted if the copy constructor is deleted, as any call to
it would make the program fail to compile.