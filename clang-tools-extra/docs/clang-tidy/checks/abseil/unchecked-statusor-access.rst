.. title:: clang-tidy - abseil-unchecked-statusor-access

abseil-unchecked-statusor-access
================================

This check identifies unsafe accesses to values contained in
``absl::StatusOr<T>`` objects. Below we will refer to this type as
``StatusOr<T>``.

An access to the value of an ``StatusOr<T>`` occurs when one of its
``value``, ``operator*``, or ``operator->`` member functions is invoked.
To align with common misconceptions, the check considers these member
functions as equivalent, even though there are subtle differences
related to exceptions vs. undefined behavior.

An access to the value of a ``StatusOr<T>`` is considered safe if and
only if code in the local scope (e.g. function body) ensures that the
status of the ``StatusOr<T>`` is ok in all possible execution paths that
can reach the access. That should happen either through an explicit
check, using the ``StatusOr<T>::ok`` member function, or by constructing
the ``StatusOr<T>`` in a way that shows that its status is unambiguously
ok (e.g. by passing a value to its constructor).

Below we list some examples of safe and unsafe ``StatusOr<T>`` access
patterns.

Note: If the check isn’t behaving as you would have expected on a code
snippet, please `report it <http://github.com/llvm/llvm-project/issues/new>`__.

False negatives
---------------

This check generally does **not** generate false negatives. That means that if
an access is not marked as unsafe, it is provably safe. If it cannot prove an
access safe, it is assumed to be unsafe. In some cases, the static analysis
cannot prove an access safe even though it is, for a variety of reasons (e.g.
unmodelled invariants of functions called). In these cases, the analysis does
produce false positive reports.

That being said, there are some heuristics used that in very rare cases might
be incorrect:

-  `a const method accessor (without arguments) that returns different
   values when called multiple times <#functionstability>`__.

If you think the check generated a false negative, please `report
it <http://github.com/llvm/llvm-project/issues/new>`__.

Known limitations
-----------------

This is a non-exhaustive list of constructs that are currently not
modelled in the check and will lead to false positives:

-  `Checking a StatusOr and then capturing it in a lambda <#lambdas>`__
-  `Indexing into a container with the same index <#containers>`__
-  `Project specific helper-functions <#uncommonapi>`__,
-  `Functions with a stable return value <#functionstability>`__
-  **Any** `cross-function reasoning <#crossfunction>`__. This is by
   design and will not change in the future.

Checking if the status is ok, then accessing the value
------------------------------------------------------

The check recognizes all straightforward ways for checking the status
and accessing the value contained in a ``StatusOr<T>`` object. For
example:

.. code:: cpp

   void f(absl::StatusOr<int> sor) {
     if (sor.ok()) {
       use(*sor);
     }
   }

Checking if the status is ok, then accessing the value from a copy
------------------------------------------------------------------

The criteria that the check uses is semantic, not syntactic. It
recognizes when a copy of the ``StatusOr<T>`` object being accessed is
known to have ok status. For example:

.. code:: cpp

   void f(absl::StatusOr<int> sor1) {
     if (sor1.ok()) {
       absl::optional<int> sor2 = sor1;
       use(*sor2);
     }
   }

Ensuring that the status is ok using common macros
--------------------------------------------------

The check is aware of common macros like ``ABSL_CHECK`` and ``ASSERT_THAT``.
Those can be used to ensure that the status of a ``StatusOr<T>`` object
is ok. For example:

.. code:: cpp

   void f(absl::StatusOr<int> sor) {
     ABSL_DCHECK_OK(sor);
     use(*sor);
   }

Ensuring that the status is ok, then accessing the value in a correlated branch
-------------------------------------------------------------------------------

The check is aware of correlated branches in the code and can figure out
when a ``StatusOr<T>`` object is ensured to have ok status on all
execution paths that lead to an access. For example:

.. code:: cpp

   void f(absl::StatusOr<int> sor) {
     bool safe = false;
     if (sor.ok() && SomeOtherCondition()) {
       safe = true;
     }
     // ... more code...
     if (safe) {
       use(*sor);
     }
   }

Accessing the value without checking the status
-----------------------------------------------

The check flags accesses to the value that are not locally guarded by a
status check:

.. code:: cpp

   void f1(absl::StatusOr<int> sor) {
     use(*sor); // unsafe: it is unclear whether the status of `sor` is ok.
   }

   void f2(absl::StatusOr<MyStruct> sor) {
     use(sor->member); // unsafe: it is unclear whether the status of `sor` is ok.
   }

   void f3(absl::StatusOr<int> sor) {
     use(sor.value()); // unsafe: it is unclear whether the status of `sor` is ok.
   }

Use ``ABSL_CHECK_OK`` to signal that you knowingly want to crash on
non-OK values.

NOTE: Even though using ``.value()``  on a non-``ok()`` ``StatusOr`` is defined
to crash, it is often unintentional. That is why our checker flags those as
well.

Accessing the value in the wrong branch
---------------------------------------

The check is aware of the state of a ``StatusOr<T>`` object in different
branches of the code. For example:

.. code:: cpp

   void f(absl::StatusOr<int> sor) {
     if (sor.ok()) {
     } else {
       use(*sor); // unsafe: it is clear that the status of `sor` is *not* ok.
     }
   }

.. _functionstability:

Assuming a function result to be stable
---------------------------------------

The check is aware that function results might not be stable. That is,
consecutive calls to the same function might return different values.
For example:

.. code:: cpp

   void f(Foo foo) {
     if (foo.sor().ok()) {
       use(*foo.sor()); // unsafe: it is unclear whether the status of `foo.sor()` is ok.
     }
   }

In such cases it is best to store the result of the function call in a
local variable and use it to access the value. For example:

.. code:: cpp

   void f(Foo foo) {
     if (const auto& foo_sor = foo.sor(); foo_sor.ok()) {
       use(*foo_sor);
     }
   }

The check **does** assume that ``const``-qualified accessor functions
return a stable value if no non-const function was called between the
two calls:

.. code:: cpp

   class Foo {
     const absl::StatusOr<int>& get() const {
       [...];
     }
   }
   void f(Foo foo) {
     if (foo.get().ok()) {
       use(*foo.get());
     }
   }

If there is a call to a non-``const``-qualified function, the check
assumes the return value of the accessor was mutated.

.. code:: cpp

   class Foo {
     const absl::StatusOr<int>& get() const {
       [...];
     }
     void mutate();
   }
   void f(Foo foo) {
     if (foo.get().ok()) {
       foo.mutate();
       use(*foo.get()); // unsafe: mutate might have changed the state of the object
     }
   }

.. _uncommonapi:

Relying on invariants of uncommon APIs
--------------------------------------

The check is unaware of invariants of uncommon APIs. For example:

.. code:: cpp

   void f(Foo foo) {
     if (foo.HasProperty("bar")) {
       use(*foo.GetProperty("bar")); // unsafe: it is unclear whether the status of `foo.GetProperty("bar")` is ok.
     }
   }

In such cases it is best to check explicitly that the status of the
``StatusOr<T>`` object is ok. For example:

.. code:: cpp

   void f(Foo foo) {
     if (const auto& property = foo.GetProperty("bar"); property.ok()) {
       use(*property);
     }
   }

.. _crossfunction:

Checking if the status is ok, then passing the ``StatusOr<T>`` to another function
----------------------------------------------------------------------------------

The check relies on local reasoning. The check and value access must
both happen in the same function. An access is considered unsafe even if
the caller of the function performing the access ensures that the status
of the ``StatusOr<T>`` is ok. For example:

.. code:: cpp

   void g(absl::StatusOr<int> sor) {
     use(*sor); // unsafe: it is unclear whether the status of `sor` is ok.
   }

   void f(absl::StatusOr<int> sor) {
     if (sor.ok()) {
       g(sor);
     }
   }

In such cases it is best to either pass the value directly when calling
a function or check that the status of the ``StatusOr<T>`` is ok in the
local scope of the callee. For example:

.. code:: cpp

   void g(int val) {
     use(val);
   }

   void f(absl::StatusOr<int> sor) {
     if (sor.ok()) {
       g(*sor);
     }
   }

Aliases created via ``using`` declarations
------------------------------------------

The check is aware of aliases of ``StatusOr<T>`` types that are created
via ``using`` declarations. For example:

.. code:: cpp

   using StatusOrInt = absl::StatusOr<int>;

   void f(StatusOrInt sor) {
     use(*sor); // unsafe: it is unclear whether the status of `sor` is ok.
   }

Containers
----------

The check is more strict than necessary when it comes to containers of
``StatusOr<T>`` values. Simply checking that the status of an element of
a container is ok is not sufficient to deem accessing it safe. For
example:

.. code:: cpp

   void f(std::vector<absl::StatusOr<int>> sors) {
     if (sors[0].ok()) {
       use(*sors[0]); // unsafe: it is unclear whether the status of `sors[0]` is ok.
     }
   }

One needs to grab a reference to a particular object and use that
instead:

.. code:: cpp

   void f(std::vector<absl::StatusOr<int>> sors) {
     absl::StatusOr<int>& sor0 = sors[0];
     if (sor0.ok()) {
       use(*sor0);
     }
   }

A future version could improve the understanding of more safe usage
patterns that involve containers.

Lambdas
-------

The check is capable of reporting unsafe ``StatusOr<T>`` accesses in
lambdas, but isn’t smart enough to propagate information from the
surrounding context through the lambda. This means that the following
pattern will be reported as an unsafe access:

.. code:: cpp

   void f(absl::StatusOr<int> sor) {
     if (sor.ok()) {
       [&sor]() {
         use(*sor); // unsafe: it is unclear whether the status of `sor` is ok.
       }
     }
   }

To avoid the issue, you should grab a reference to the contained object
and capture that instead

.. code:: cpp

   void f(absl::StatusOr<int> sor) {
     if (sor.ok()) {
       auto& s = *sor;
       [&s]() {
         use(s);
       }
     }
   }

Alternatively you could add a check inside the lambda where the value is
accessed:

.. code:: cpp

   void f(absl::StatusOr<int> sor) {
     [&sor]() {
       if (sor.ok()) {
         use(*sor);
       }
     }
   }
