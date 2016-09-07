.. title:: clang-tidy - modernize-pass-by-value

modernize-pass-by-value
=======================

With move semantics added to the language and the standard library updated with
move constructors added for many types it is now interesting to take an
argument directly by value, instead of by const-reference, and then copy. This
check allows the compiler to take care of choosing the best way to construct
the copy.

The transformation is usually beneficial when the calling code passes an
*rvalue* and assumes the move construction is a cheap operation. This short
example illustrates how the construction of the value happens:

  .. code-block:: c++

    void foo(std::string s);
    std::string get_str();

    void f(const std::string &str) {
      foo(str);       // lvalue  -> copy construction
      foo(get_str()); // prvalue -> move construction
    }

.. note::

   Currently, only constructors are transformed to make use of pass-by-value.
   Contributions that handle other situations are welcome!


Pass-by-value in constructors
-----------------------------

Replaces the uses of const-references constructor parameters that are copied
into class fields. The parameter is then moved with `std::move()`.

Since ``std::move()`` is a library function declared in `<utility>` it may be
necessary to add this include. The check will add the include directive when
necessary.

  .. code-block:: c++

     #include <string>

     class Foo {
     public:
    -  Foo(const std::string &Copied, const std::string &ReadOnly)
    -    : Copied(Copied), ReadOnly(ReadOnly)
    +  Foo(std::string Copied, const std::string &ReadOnly)
    +    : Copied(std::move(Copied)), ReadOnly(ReadOnly)
       {}

     private:
       std::string Copied;
       const std::string &ReadOnly;
     };

     std::string get_cwd();

     void f(const std::string &Path) {
       // The parameter corresponding to 'get_cwd()' is move-constructed. By
       // using pass-by-value in the Foo constructor we managed to avoid a
       // copy-construction.
       Foo foo(get_cwd(), Path);
     }


If the parameter is used more than once no transformation is performed since
moved objects have an undefined state. It means the following code will be left
untouched:

.. code-block:: c++

  #include <string>

  void pass(const std::string &S);

  struct Foo {
    Foo(const std::string &S) : Str(S) {
      pass(S);
    }

    std::string Str;
  };


Known limitations
^^^^^^^^^^^^^^^^^

A situation where the generated code can be wrong is when the object referenced
is modified before the assignment in the init-list through a "hidden" reference.

Example:

.. code-block:: c++

   std::string s("foo");

   struct Base {
     Base() {
       s = "bar";
     }
   };

   struct Derived : Base {
  -  Derived(const std::string &S) : Field(S)
  +  Derived(std::string S) : Field(std::move(S))
     { }

     std::string Field;
   };

   void f() {
  -  Derived d(s); // d.Field holds "bar"
  +  Derived d(s); // d.Field holds "foo"
   }


Note about delayed template parsing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When delayed template parsing is enabled, constructors part of templated
contexts; templated constructors, constructors in class templates, constructors
of inner classes of template classes, etc., are not transformed. Delayed
template parsing is enabled by default on Windows as a Microsoft extension:
`Clang Compiler User’s Manual - Microsoft extensions`_.

Delayed template parsing can be enabled using the `-fdelayed-template-parsing`
flag and disabled using `-fno-delayed-template-parsing`.

Example:

.. code-block:: c++

   template <typename T> class C {
     std::string S;

   public:
 =  // using -fdelayed-template-parsing (default on Windows)
 =  C(const std::string &S) : S(S) {}
 
 +  // using -fno-delayed-template-parsing (default on non-Windows systems)
 +  C(std::string S) : S(std::move(S)) {}
   };

.. _Clang Compiler User’s Manual - Microsoft extensions: http://clang.llvm.org/docs/UsersManual.html#microsoft-extensions

.. seealso::

  For more information about the pass-by-value idiom, read: `Want Speed? Pass by Value`_.

  .. _Want Speed? Pass by Value: http://cpp-next.com/archive/2009/08/want-speed-pass-by-value/

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.
