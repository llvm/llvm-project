.. title:: clang-tidy - cppcoreguidelines-avoid-non-const-global-variables

cppcoreguidelines-avoid-non-const-global-variables
==================================================

Finds non-const global variables as described in `I.2
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#i2-avoid-non-const-global-variables>`_
of C++ Core Guidelines.
As `R.6 <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rr-global>`_
of C++ Core Guidelines is a duplicate of rule `I.2
<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#i2-avoid-non-const-global-variables>`_
it also covers that rule.

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

The variables ``a``, ``c``, ``c_ptr1``, ``c_const_ptr`` and ``c_reference``
will all generate warnings since they are either a non-const globally accessible
variable, a pointer or a reference providing global access to non-const data
or both.

Options
-------

.. option:: AllowInternalLinkage

   When set to `true`, static non-const variables and variables in anonymous
   namespaces will not generate a warning. The default value is `false`.
