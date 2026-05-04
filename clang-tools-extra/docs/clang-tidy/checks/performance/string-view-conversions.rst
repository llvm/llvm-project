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
        foo(42, std::string(sv).c_str(), 2.71); // conversion to std::string and
                                                // then to char* is redundant
                                                // as std::string_view
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
        foo(42, sv, 2.71);
        foo(42, "foo", 3.14);
    }

Please note
-----------

Pattern ``std::string(sv).c_str()`` can be used intentionally to copy
the given string up to the null byte. If so, you may use ``NOLINT``
or rewrite your code:

Before:

.. code-block:: c++

    void foo(int p1, std::string_view p2, double p3);
    void bar(std::string_view sv) {
        foo(42, std::string(sv).c_str(), 2.71); // warning is emitted
    }

After:

.. code-block:: c++

    void foo(int p1, std::string_view p2, double p3);
    void bar(std::string_view sv) {
        std::string s(sv);
        // No warning emitted for the next 2 lines
        foo(42, s.c_str(), 2.71); // explicit std::string variable used
        foo(42, std::string(sv).c_str(), 2.71); // NOLINT(performance-string-view-conversions)
    }
