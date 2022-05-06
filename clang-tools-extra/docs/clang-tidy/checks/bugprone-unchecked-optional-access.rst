.. title:: clang-tidy - bugprone-unchecked-optional-access

bugprone-unchecked-optional-access
==================================

*Note*: This check uses a flow-sensitive static analysis to produce its
 results. Therefore, it may be more resource intensive (RAM, CPU) than the
 average clang-tidy check.

This check identifies unsafe accesses to values contained in
``std::optional<T>``, ``absl::optional<T>``, or ``base::Optional<T>``
objects. Below we will refer to all these types collectively as
``optional<T>``.

An access to the value of an ``optional<T>`` occurs when one of its
``value``, ``operator*``, or ``operator->`` member functions is invoked.
To align with common misconceptions, the check considers these member
functions as equivalent, even though there are subtle differences
related to exceptions versus undefined behavior. See
go/optional-style-recommendations for more information on that topic.

An access to the value of an ``optional<T>`` is considered safe if and only if
code in the local scope (for example, a function body) ensures that the
``optional<T>`` has a value in all possible execution paths that can reach the
access. That should happen either through an explicit check, using the
``optional<T>::has_value`` member function, or by constructing the
``optional<T>`` in a way that shows that it unambiguously holds a value (e.g
using ``std::make_optional`` which always returns a populated
``std::optional<T>``).

Below we list some examples, starting with unsafe optional access patterns,
followed by safe access patterns.

Unsafe access patterns
~~~~~~~~~~~~~~~~~~~~~~

Access the value without checking if it exists
----------------------------------------------

The check flags accesses to the value that are not locally guarded by
existence check:

.. code-block:: c++

   void f(absl::optional<int> opt) {
     use(*opt); // unsafe: it is unclear whether `opt` has a value.
   }

Access the value in the wrong branch
------------------------------------

The check is aware of the state of an optional object in different
branches of the code. For example:

.. code-block:: c++

   void f(absl::optional<int> opt) {
     if (opt.has_value()) {
     } else {
       use(opt.value()); // unsafe: it is clear that `opt` does *not* have a value.
     }
   }

Assume a function result to be stable
-------------------------------------

The check is aware that function results might not be stable. That is,
consecutive calls to the same function might return different values.
For example:

.. code-block:: c++

   void f(Foo foo) {
     if (foo.opt().has_value()) {
       use(*foo.opt()); // unsafe: it is unclear whether `foo.opt()` has a value.
     }
   }

Rely on invariants of uncommon APIs
-----------------------------------

The check is unaware of invariants of uncommon APIs. For example:

.. code-block:: c++

   void f(Foo foo) {
     if (foo.HasProperty("bar")) {
       use(*foo.GetProperty("bar")); // unsafe: it is unclear whether `foo.GetProperty("bar")` has a value.
     }
   }

Check if a value exists, then pass the optional to another function
-------------------------------------------------------------------

The check relies on local reasoning. The check and value access must
both happen in the same function. An access is considered unsafe even if
the caller of the function performing the access ensures that the
optional has a value. For example:

.. code-block:: c++

   void g(absl::optional<int> opt) {
     use(*opt); // unsafe: it is unclear whether `opt` has a value.
   }

   void f(absl::optional<int> opt) {
     if (opt.has_value()) {
       g(opt);
     }
   }

Safe access patterns
~~~~~~~~~~~~~~~~~~~~

Check if a value exists, then access the value
----------------------------------------------

The check recognizes all straightforward ways for checking if a value
exists and accessing the value contained in an optional object. For
example:

.. code-block:: c++

   void f(absl::optional<int> opt) {
     if (opt.has_value()) {
       use(*opt);
     }
   }


Check if a value exists, then access the value from a copy
----------------------------------------------------------

The criteria that the check uses is semantic, not syntactic. It
recognizes when a copy of the optional object being accessed is known to
have a value. For example:

.. code-block:: c++

   void f(absl::optional<int> opt1) {
     if (opt1.has_value()) {
       absl::optional<int> opt2 = opt1;
       use(*opt2);
     }
   }


Ensure that a value exists using common macros
----------------------------------------------

The check is aware of common macros like ``CHECK``, ``DCHECK``, and
``ASSERT_THAT``. Those can be used to ensure that an optional object has
a value. For example:

.. code-block:: c++

   void f(absl::optional<int> opt) {
     DCHECK(opt.has_value());
     use(*opt);
   }

Ensure that a value exists, then access the value in a correlated branch
------------------------------------------------------------------------

The check is aware of correlated branches in the code and can figure out
when an optional object is ensured to have a value on all execution
paths that lead to an access. For example:

.. code-block:: c++

   void f(absl::optional<int> opt) {
     bool safe = false;
     if (opt.has_value() && SomeOtherCondition()) {
       safe = true;
     }
     // ... more code...
     if (safe) {
       use(*opt);
     }
   }

Stabilize function results
~~~~~~~~~~~~~~~~~~~~~~~~~~

Since function results are not assumed to be stable across calls, it is best to
store the result of the function call in a local variable and use that variable
to access the value. For example:

.. code-block:: c++

   void f(Foo foo) {
     if (const auto& foo_opt = foo.opt(); foo_opt.has_value()) {
       use(*foo_opt);
     }
   }

Do not rely on uncommon-API invariants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When uncommon APIs guarantee that an optional has contents, do not rely on it --
instead, check explicitly that the optional object has a value. For example:

.. code-block:: c++

   void f(Foo foo) {
     if (const auto& property = foo.GetProperty("bar")) {
       use(*property);
     }
   }

instead of the `HasProperty`, `GetProperty` pairing we saw above.

Do not rely on caller-performed checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you know that all of a function's callers have checked that an optional
argument has a value, either change the function to take the value directly or
check the optional again in the local scope of the callee. For example:

.. code-block:: c++

   void g(int val) {
     use(val);
   }

   void f(absl::optional<int> opt) {
     if (opt.has_value()) {
       g(*opt);
     }
   }

and

.. code-block:: c++

   struct S {
     absl::optional<int> opt;
     int x;
   };

   void g(const S &s) {
     if (s.opt.has_value() && s.x > 10) {
       use(*s.opt);
   }

   void f(S s) {
     if (s.opt.has_value()) {
       g(s);
     }
   }

Additional notes
~~~~~~~~~~~~~~~~

Aliases created via ``using`` declarations
------------------------------------------

The check is aware of aliases of optional types that are created via
``using`` declarations. For example:

.. code-block:: c++

   using OptionalInt = absl::optional<int>;

   void f(OptionalInt opt) {
     use(opt.value()); // unsafe: it is unclear whether `opt` has a value.
   }

Lambdas
-------

The check does not currently report unsafe optional acceses in lambdas.
A future version will expand the scope to lambdas, following the rules
outlined above. It is best to follow the same principles when using
optionals in lambdas.
