.. title:: clang-tidy - modernize-replace-auto-ptr

modernize-replace-auto-ptr
==========================

This check replaces the uses of the deprecated class ``std::auto_ptr`` by
``std::unique_ptr`` (introduced in C++11). The transfer of ownership, done
by the copy-constructor and the assignment operator, is changed to match
``std::unique_ptr`` usage by using explicit calls to ``std::move()``.

Migration example:

.. code-block:: c++

  -void take_ownership_fn(std::auto_ptr<int> int_ptr);
  +void take_ownership_fn(std::unique_ptr<int> int_ptr);

   void f(int x) {
  -  std::auto_ptr<int> a(new int(x));
  -  std::auto_ptr<int> b;
  +  std::unique_ptr<int> a(new int(x));
  +  std::unique_ptr<int> b;

  -  b = a;
  -  take_ownership_fn(b);
  +  b = std::move(a);
  +  take_ownership_fn(std::move(b));
   }

Since ``std::move()`` is a library function declared in ``<utility>`` it may be
necessary to add this include. The check will add the include directive when
necessary.

Known Limitations
-----------------
* If headers modification is not activated or if a header is not allowed to be
  changed this check will produce broken code (compilation error), where the
  headers' code will stay unchanged while the code using them will be changed.

* Client code that declares a reference to an ``std::auto_ptr`` coming from
  code that can't be migrated (such as a header coming from a 3\ :sup:`rd`
  party library) will produce a compilation error after migration. This is
  because the type of the reference will be changed to ``std::unique_ptr`` but
  the type returned by the library won't change, binding a reference to
  ``std::unique_ptr`` from an ``std::auto_ptr``. This pattern doesn't make much
  sense and usually ``std::auto_ptr`` are stored by value (otherwise what is
  the point in using them instead of a reference or a pointer?).

  .. code-block:: c++

     // <3rd-party header...>
     std::auto_ptr<int> get_value();
     const std::auto_ptr<int> & get_ref();

     // <calling code (with migration)...>
    -std::auto_ptr<int> a(get_value());
    +std::unique_ptr<int> a(get_value()); // ok, unique_ptr constructed from auto_ptr

    -const std::auto_ptr<int> & p = get_ptr();
    +const std::unique_ptr<int> & p = get_ptr(); // won't compile

* Non-instantiated templates aren't modified.

  .. code-block:: c++

     template <typename X>
     void f() {
         std::auto_ptr<X> p;
     }

     // only 'f<int>()' (or similar) will trigger the replacement.

Options
-------

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.
