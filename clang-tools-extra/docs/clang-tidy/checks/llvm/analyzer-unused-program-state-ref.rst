.. title:: clang-tidy - llvm-analyzer-unused-program-state-ref

llvm-analyzer-unused-program-state-ref
======================================

Finds unused local ``clang::ento::ProgramStateRef`` variables in Clang Static
Analyzer code.

``ProgramStateRef`` is an ``IntrusiveRefCntPtr<const ProgramState>``, i.e. a
reference-counted smart pointer with a non-trivial destructor. Because of that
destructor, ``-Wunused-variable`` never reports such a variable, even when it is
declared and never read -- the compiler conservatively assumes the destructor
might have side effects. ``ProgramStateRef`` carries no meaningful RAII side
effects, so an unused one is simply dead code, usually left behind after a
refactoring.

.. code-block:: c++

  void Checker::checkPreCall(const CallEvent &Call, CheckerContext &C) const {
    ProgramStateRef State = C.getState();
    // ... State is never used ...
  }

The declaration above is dead code and can be deleted by hand. For a statement
that declares several variables at once (``ProgramStateRef A, B;``), each unused
declarator is reported individually; if only some are unused the others are left
untouched. A declaration that originates from a macro expansion is still
reported.

Structured bindings are also handled, e.g.

.. code-block:: c++

  auto [StTrue, StFalse] = State->assume(V); // both unused

is reported. A structured binding is only flagged when *every* binding is an
unused ``ProgramStateRef``; a partially-used decomposition is left alone.
