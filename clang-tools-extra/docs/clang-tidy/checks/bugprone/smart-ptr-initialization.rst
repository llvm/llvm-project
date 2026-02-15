.. title:: clang-tidy - bugprone-smart-ptr-initialization

bugprone-smart-ptr-initialization
==================================

Detects dangerous initialization of smart pointers with raw pointers that are
already owned elsewhere, which can lead to double deletion.

This check implements CERT C++ rule `MEM56-CPP. Do not store an already-owned
pointer value in an unrelated smart pointer
<https://wiki.sei.cmu.edu/confluence/display/cplusplus/MEM56-CPP.+Do+not+store+an+already-owned+pointer+value+in+an+unrelated+smart+pointer>`_.

Examples
--------

The check flags cases where raw pointers that are already owned or managed
elsewhere are passed to smart pointer constructors or ``reset()`` methods:

.. code-block:: c++

  A& getA();
  void foo() {
    // Warning: '&getA()' is already managed elsewhere
    std::shared_ptr<A> a(&getA());
  }

  void bar() {
    int x = 10;
    // Warning: '&x' points to a local variable
    std::unique_ptr<int> ptr(&x);
  }

  void baz() {
    std::vector<int> vec{1, 2, 3};
    std::shared_ptr<int> sp;
    // Warning: '&vec[0]' is managed by the vector
    sp.reset(&vec[0]);
  }

Allowed cases
-------------

The check ignores legitimate cases:

1. **New expressions**: Pointers from ``new`` operators are safe:

   .. code-block:: c++

     std::unique_ptr<int> p(new int(5));  // OK

2. **Release calls**: Pointers from ``release()`` method are transferred:

   .. code-block:: c++

     auto p1 = std::make_unique<int>(5);
     std::unique_ptr<int> p2(p1.release());  // OK

3. **Custom deleters**: Smart pointers with custom deleters are ignored:

   .. code-block:: c++

     void customDeleter(int* p) { delete p; }
     std::unique_ptr<int, decltype(&customDeleter)> p(&getA(), customDeleter);

4. **Null pointers**: ``nullptr`` is always safe:

   .. code-block:: c++

     std::shared_ptr<int> p(nullptr);  // OK
     p.reset(nullptr);  // OK

Options
-------

.. option:: SharedPointers

   A semicolon-separated list of (fully qualified) shared pointer type names
   that should be checked. Default value is
   `::std::shared_ptr;::boost::shared_ptr`.

.. option:: UniquePointers

   A semicolon-separated list of (fully qualified) unique pointer type names
   that should be checked. Default value is
   `::std::unique_ptr`.

.. option:: DefaultDeleters

   A semicolon-separated list of (fully qualified) default deleter type names.
   Smart pointers with deleters matching these types are considered to use the
   default deleter and are checked. Smart pointers with custom deleters are
   ignored. Default value is `::std::default_delete`.

Limitations
----------

This check only supports smart pointers with shared and unique ownership
semantics. Smart pointers with different semantics, such as
``boost::scoped_ptr``, cannot be used with the current version of this check.

References
----------

* `CERT C++ MEM56-CPP <https://wiki.sei.cmu.edu/confluence/display/cplusplus/MEM56-CPP.+Do+not+store+an+already-owned+pointer+value+in+an+unrelated+smart+pointer>`_
* `C++ Core Guidelines R.3: A raw pointer (a T*) is non-owning <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#r3-a-raw-pointer-a-t-is-non-owning>`_
* `C++ Core Guidelines R.20: Use unique_ptr or shared_ptr to represent ownership <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#r20-use-unique_ptr-or-shared_ptr-to-represent-ownership>`_
