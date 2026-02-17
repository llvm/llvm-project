.. title:: clang-tidy - performance-string-view-conversions

performance-string-view-conversions
===================================

Finds and removes redundant conversions from ``std::[w|u8|u16|u32]string_view``
to ``std::[...]string`` in call expressions expecting ``std::[...]string_view``.

Before:

.. code-block:: c++

    void foo(int p1, std::string_view p2, double p3);
    void bar(std::string_view sv) {
        foo(42, std::string(sv), 3.14); // conversion to std::string is
                                        // redundant as std::string_view
                                        // is expected
        foo(42, std::string("foo"), 3.14); // conversion to std::string is
                                           // redundant as std::string_view
                                           // is expected
    }

After:

.. code-block:: c++

    void foo(int p1, std::string_view p2, double p3);
    void bar(std::string_view sv) {
        foo(42, sv, 3.14);
        foo(42, "foo", 3.14);
    }
