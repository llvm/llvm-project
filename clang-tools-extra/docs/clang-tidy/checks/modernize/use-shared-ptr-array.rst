.. title:: clang-tidy - modernize-use-shared-ptr-array

modernize-use-shared-ptr-array
==============================

The check requires C++17 or later.

Recognized Deleter Forms
------------------------

.. code-block:: c++

   // std::shared_ptr<Foo> p(new Foo[10], std::default_delete<Foo[]>());
   // std::shared_ptr<Foo> p(new Foo[10], std::default_delete<Foo[]>{});
   std::shared_ptr<Foo[]> p(new Foo[10]);

   // using FooDeleter = std::default_delete<Foo[]>;
   // std::shared_ptr<Foo> p(new Foo[10], FooDeleter());
   std::shared_ptr<Foo[]> p(new Foo[10]);

   // namespace std2 = std;
   // std::shared_ptr<Foo> p(new Foo[10], std2::default_delete<Foo[]>());
   // std::shared_ptr<Foo> p(new Foo[10], ::std::default_delete<Foo[]>());
   std::shared_ptr<Foo[]> p(new Foo[10]);

   // std::shared_ptr<Foo> p(new Foo[10], [](Foo *p) { delete[] p; });
   std::shared_ptr<Foo[]> p(new Foo[10]);

   // void deleteFoo(Foo *p) { delete[] p; }
   // std::shared_ptr<Foo> p(new Foo[10], deleteFoo);
   std::shared_ptr<Foo[]> p(new Foo[10]);
