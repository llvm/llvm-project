===========================
C++ Profiles Framework
===========================

.. contents::
   :depth: 3
   :local:


Introduction
============

The C++ Profiles framework (`P3589R2
<https://open-std.org/JTC1/SC22/WG21/docs/papers/2025/p3589r2.pdf>`_) allows a
translation unit to opt into additional language restrictions called *profiles*.
A profile is a named set of rules enforced by the compiler. A translation unit
requests enforcement with ``[[profiles::enforce(...)]]``; individual
declarations or statements can suppress enforcement with
``[[profiles::suppress(...)]]``; and module imports can require that an imported
module enforces a profile with ``[[profiles::require(...)]]``.

Profiles do not change the meaning of well-formed programs with no undefined
behavior.  Their static semantic effects are conceptually applied only after
translation phase 7: a profile cannot change the outcome of overload resolution
or template instantiation, and it is not possible to SFINAE on a profile
violation.

The framework is profile-agnostic.  It handles attribute parsing, enforcement
tracking, suppression scoping, module integration, and serialization.
Individual profiles only need to call a single API
(``Sema::checkProfileViolation``) at the appropriate semantic check sites.
Everything else -- suppression, template instantiation, SFINAE exclusion,
module propagation, and PCH/BMI serialization -- is handled by the framework
automatically.

The entire framework is gated on the ``-fprofiles`` command-line flag
(``LangOpts.Profiles``).


Implementing a New Profile
==========================

Adding a new profile requires no changes to the framework itself.  A profile is
defined entirely by:

1. A **profile name** (a ``::``-separated identifier sequence such as
   ``vendor::safety`` or ``std::type``).
2. One or more **rule names** (string identifiers such as
   ``"reinterpret_cast"``).
3. **Diagnostics** emitted when a rule is violated.
4. **Check sites** in the compiler where ``checkProfileViolation`` is called.

There is no central registry of profiles.  The framework treats profile names as
opaque strings; a profile is considered "enforced" simply because the user wrote
``[[profiles::enforce(name)]]`` in their source.  Each rule within a profile is
likewise just a string identifier; users can suppress individual rules with
``[[profiles::suppress(profile_name, rule: "rule_name")]]``.

Define Diagnostics
------------------

Add diagnostics to ``clang/include/clang/Basic/DiagnosticSemaKinds.td`` in the
``// C++ Profiles framework (P3589R2)`` group.  Each diagnostic should accept
``%0`` for the profile name, since ``checkProfileViolation`` passes the profile
name as the first diagnostic argument.

For example, the ``test::type_cast`` profile defines:

.. code-block:: text

   def err_profile_type_cast_reinterpret : Error<
     "'reinterpret_cast' is unsafe under profile '%0'">;

Add ``checkProfileViolation`` Calls
------------------------------------

At each semantic site where a rule can be violated, call
``Sema::checkProfileViolation``:

.. code-block:: c++

   checkProfileViolation("my::profile", "my_rule", Loc,
                         diag::err_my_profile_rule);

This function checks whether the profile is enforced and not suppressed.  During
template argument deduction, profile rule diagnostics are suppressed by Clang's
normal SFINAE machinery so they cannot affect overload resolution or template
instantiation; selected specializations replay suppressed diagnostics when used.
Unevaluated and discarded-statement contexts are skipped.  The profile name is
passed as ``%0``.


Framework Internals Reference
=============================

This section describes the framework mechanisms that profile implementers
benefit from understanding, even though they do not interact with them directly.

``ProfileSuppressScope``
------------------------

An RAII guard that pushes suppression entries onto ``Sema::ProfileSuppressStack``
and pops them on destruction.  It is used by the parser and template
instantiation machinery to make ``[[profiles::suppress]]`` attributes active
during the appropriate region.  ``checkProfileViolation`` consults
``ProfileSuppressStack`` directly, so profile implementers never need to create
``ProfileSuppressScope`` objects.

Template Instantiation
----------------------

During template instantiation, the framework ensures that
``[[profiles::suppress]]`` on the template pattern and its lexical parents
applies to instantiated code.  This is done via ``ProfileSuppressScope`` with
``WalkLexicalParents=true`` at several sites:

- ``SemaTemplateInstantiateDecl.cpp`` -- function and variable template
  instantiation.
- ``SemaTemplateInstantiate.cpp`` -- default member initializer instantiation.
- ``TreeTransform.h`` -- ``TransformAttributedStmt`` (suppress on statements)
  and ``TransformDeclStmt`` (suppress on declarations within a ``DeclStmt``).

Module Enforcement
------------------

``[[profiles::enforce(...)]]`` on a module interface declaration records the
enforced profile designators on ``Module::EnforcedProfileDesignators``.  Module
implementation units automatically inherit the interface's enforcements.
``[[profiles::require(...)]]`` on an import-declaration validates that the
imported module's ``EnforcedProfileDesignators`` contains a matching designator.

Importing a module that enforces a profile does **not** enforce that profile in
the importing translation unit.  Enforcement is always explicit and local.

Serialization
-------------

The framework serializes enforcement state automatically.  Profile implementers
do not need to add any serialization code.

- **PCH**: ``Sema::EnforcedProfiles`` is written as ``ENFORCED_PROFILES``
  records in the AST bitstream and restored when the PCH is loaded.
- **Module BMI**: ``Module::EnforcedProfileDesignators`` is written as
  ``SUBMODULE_ENFORCED_PROFILES`` records within each submodule block.

The ``test::type_cast`` Profile
===============================

The ``test::type_cast`` profile is a minimal, test-only profile included in the
tree.  It demonstrates everything needed to implement a profile.

The profile has a single rule, ``reinterpret_cast``, which diagnoses uses of
``reinterpret_cast``.  In ``clang/lib/Sema/SemaCast.cpp``, inside the
``reinterpret_cast`` handling of ``Sema::BuildCXXNamedCast``:

.. code-block:: c++

   checkProfileViolation("test::type_cast", "reinterpret_cast", OpLoc,
                         diag::err_profile_type_cast_reinterpret);

That single call is the entire profile implementation.
