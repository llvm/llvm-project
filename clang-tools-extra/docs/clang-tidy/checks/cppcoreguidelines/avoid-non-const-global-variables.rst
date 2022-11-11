.. title:: clang-tidy - cppcoreguidelines-avoid-non-const-global-variables

cppcoreguidelines-avoid-non-const-global-variables
==================================================

Finds non-const global variables as described in `I.2 of C++ Core Guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Ri-global>`_ .
As `R.6 of C++ Core Guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rr-global>`_ is a duplicate of rule I.2 it also covers that rule.

.. code-block:: c++

    char a;  // Warns!
    const char b =  0;

    namespace some_namespace
    {
        char c;  // Warns!
        const char d = 0;
    }

    char * c_ptr1 = &some_namespace::c;  // Warns!
    char *const c_const_ptr = &some_namespace::c;  // Warns!
    char & c_reference = some_namespace::c;  // Warns!

    class Foo  // No Warnings inside Foo, only namespace scope is covered
    {
    public:
        char e = 0;
        const char f = 0;
    protected:
        char g = 0;
    private:
        char h = 0;
    };

Variables: ``a``, ``c``, ``c_ptr1``, ``c_ptr2``, ``c_const_ptr`` and
``c_reference``, will all generate warnings since they are either:
a globally accessible variable and non-const, a pointer or reference providing
global access to non-const data or both.
