.. title:: clang-tidy - bugprone-null-check-after-dereference

bugprone-null-check-after-dereference
=====================================

.. note::

   This check uses a flow-sensitive static analysis to produce its
   results. Therefore, it may be more resource intensive (RAM, CPU) than the
   average Clang-tidy check.

This check identifies redundant pointer null-checks, by finding cases where the
pointer cannot be null at the location of the null-check.

Redundant null-checks can signal faulty assumptions about the current value of
a pointer at different points in the program. Either the null-check is
redundant, or there could be a null-pointer dereference earlier in the program.

.. code-block:: c++

   int f(int *ptr) {
     *ptr = 20; // note: one of the locations where the pointer's value cannot be null
     // ...
     if (ptr) { // bugprone: pointer is checked even though it cannot be null at this point
       return *ptr;
     }
     return 0;
   }

Supported pointer operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pointer null-checks
-------------------

The check currently supports null-checks on pointers that use
``operator bool``, such as when being used as the condition
for an ``if`` statement. It also supports comparisons such as ``!= nullptr``, and
``== other_ptr``.

.. code-block:: c++

   int f(int *ptr) {
     if (ptr) {
       if (ptr) { // bugprone: pointer is re-checked after its null-ness is already checked.
         return *ptr;
       }

       return ptr ? *ptr : 0; // bugprone: pointer is re-checked after its null-ness is already checked.
     }
     return 0;
   }

Pointer dereferences
--------------------

Pointer star- and arrow-dereferences are supported.

.. code-block:: c++

   struct S {
     int val;
   };

   void f(int *ptr, S *wrapper) {
     *ptr = 20;
     wrapper->val = 15;
   }

Null-pointer and other value assignments
----------------------------------------

The check supports assigning various values to pointers, making them *null*
or *non-null*. The check also supports passing pointers of a pointer to
external functions.

.. code-block:: c++

   extern int *external();
   extern void refresh(int **ptr_ptr);
   
   int f() {
     int *ptr_null = nullptr;
     if (ptr_null) { // bugprone: pointer is checked where it cannot be non-null.
       return *ptr_null;
     }

     int *ptr = external();
     if (ptr) { // safe: external() could return either nullable or nonnull pointers.
       return *ptr;
     }

     int *ptr2 = external();
     *ptr2 = 20;
     refresh(&ptr2);
     if (ptr2) { // safe: pointer could be changed by refresh().
       return *ptr2;
     }
     return 0;
   }

Limitations
~~~~~~~~~~~

The check only supports C++ due to limitations in the data-flow framework.

The annotations ``_Nullable`` and ``_Nonnull`` are not supported.

.. code-block:: c++

   extern int *_nonnull external_nonnull();

   int annotations() {
     int *ptr = external_nonnull();

     return ptr ? *ptr : 0; // false-negative: pointer is known to be non-null.
   }

Function calls taking a pointer value as a reference or a pointer-to-pointer are
not supported.

.. code-block:: c++

   extern int *external();
   extern void refresh_ref(int *&ptr);
   extern void refresh_ptr(int **ptr);

   int extern_ref() {
     int *ptr = external();
     *ptr = 20;

     refresh_ref(ptr);
     refresh_ptr(&ptr);

     return ptr ? *ptr : 0; // false-positive: pointer could be changed by refresh_ref().
   }

Note tags are currently appended to a single location, even if all paths ensure
a pointer is not null.

.. code-block:: c++

   int branches(int *p, bool b) {
     if (b) {
       *p = 42; // true-positive: note-tag appended here
     } else {
       *p = 20; // false-positive: note tag not appended here
     }

     return ptr ? *ptr : 0;
   }

Declarations and some other operations are not supported by note tags yet. This
can sometimes result in erroneous note tags being shown instead of the correct
one.

.. code-block:: c++

   int note_tags() {
      int *ptr = nullptr; // false-negative: note tag not shown

      return ptr ? *ptr : 0;
   }
