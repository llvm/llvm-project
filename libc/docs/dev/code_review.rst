.. _code_review:

=============================
Reviewing LLVM-libc Patches
=============================

This page describes what reviewers look for in LLVM-libc pull requests. It is a
libc-specific supplement to the project-wide
`LLVM Code-Review Policy <https://llvm.org/docs/CodeReview.html>`__ and
`LLVM Developer Policy <https://llvm.org/docs/DeveloperPolicy.html>`__, **not a
replacement for them**. The mechanics that apply to every LLVM subproject are 
defined there and are assumed here.

The goal of this document is narrower: to make the *libc-specific* expectations
explicit so that both reviewers and contributors know what tends to block a libc
patch, independent of general LLVM review practice.

.. note::

   Nothing here grants approval authority or overrides the LLVM-wide policy. When
   this page and the LLVM-wide policy appear to differ, the LLVM-wide policy
   governs the process; this page only adds libc-specific technical checks.

Scope of a Reviewable Patch
===========================

LLVM-libc patches are expected to be small and single-purpose. A reviewer should
push back when a PR mixes unrelated concerns for example, a new entrypoint
bundled with an unrelated style cleanup, or several independent bug fixes in one
diff. The community norm is to land each logical change as its own PR with its
own test, even when the changes are individually trivial; bundling makes
bisection, reverting, and attribution harder.

When reviewing, check that:

* the PR does one thing, and the description says what and why;
* every behavioral change is accompanied by a test in the same PR (see
  `Testing`_);
* a new entrypoint comes with its build-system registration, header-generation
  entry, and configuration updates (see `Entrypoint Mechanics`_), not as a
  follow-up.

Entrypoint Mechanics
====================

Most libc patches add or modify an *entrypoint* (a public function or global
variable). Reviewers verify that the patch follows the entrypoint conventions
described in :ref:`implementation_standard` and :ref:`entrypoints`. Common
things to check:

* **Definition macro.** The function is defined with the ``LLVM_LIBC_FUNCTION``
  macro (and variables with ``LLVM_LIBC_VARIABLE``), so that the C alias symbol
  is produced correctly. A plain C++ definition with no macro is a defect.
* **Namespace.** All implementation constructs are inside ``LIBC_NAMESPACE_DECL``.
* **Header layout.** There is an internal implementation header
  (``src/<header>/<name>.h``) declaring the function in the namespace, paired
  with the ``.cpp``, placed under the directory matching the public header it
  belongs to (see :ref:`source_tree_layout`). Platform-specific implementations
  live in the appropriate platform subdirectory (e.g.
  ``src/stdio/linux/remove.cpp``).
* **Build registration.** The entrypoint is registered with
  ``add_entrypoint_object`` and lists its real dependencies. Missing or
  over-broad dependency lists should be flagged.
* **Configuration.** The target is wired into the relevant
  ``config/<platform>/<arch>/entrypoints.txt`` (and ``headers.txt`` /
  ``exclude.txt`` where applicable) for every platform the change claims to
  support -- not just the contributor's host.
* **Header generation.** The public-header spec is updated so the declaration is
  generated, rather than hand-edited into a generated header.

Build-System Completeness
=========================

A patch that builds for the author can break another build mode Before approving,
confirm the patch updates **all** of the build surfaces it touches:

* **CMake** -- the primary build, including ``add_entrypoint_object`` /
  ``add_object_library`` targets and test registration.
* **Bazel overlay** (``utils/bazel/llvm-project-overlay/libc/``) -- the Bazel
  build is not optional. Patches that add entrypoints or move files but skip the
  Bazel overlay routinely require a separate ``[Bazel] Fixes`` follow-up; ask for
  the overlay update in the same PR.
* **Per-platform config** -- as above, the ``config/`` entrypoint lists for each
  affected target.

CI covers much of this, but reviewers should not rely on a contributor's host
configuration to exercise GPU, baremetal, or non-host targets that the change
claims to support.

Warnings are treated as defects. Code is expected to compile cleanly, without
warnings, on the minimum supported compilers; a patch that introduces new
compiler or clang-tidy warnings should be fixed before approval rather than
left for a follow-up.

Code Style and Internal Reuse
=============================

Beyond the project-wide LLVM style, libc has its own conventions documented in
:ref:`code_style`. Reviewers commonly flag:

* **Naming.** ``snake_case`` for functions and non-const variables,
  ``SNAKE_CASE`` for ``const``/``constexpr`` values, ``CapitalizedCamelCase``
  for internal types, and standard-prescribed names for public symbols.
* **Inlining of internal helpers.** Internal helper functions in headers should
  be marked ``LIBC_INLINE`` so the entrypoint does not pay call overhead for
  them except where genuinely required (for example, real recursion).
* **Reuse of** ``__support``. New code should reuse the internal utilities under
  ``src/__support`` (e.g. the ``cpp::`` standard-library shims, ``FPBits`` and
  the floating-point helpers, integer and bit-manipulation utilities, the
  ``CPP`` containers) rather than re-implementing them or reaching for ``std::``.
* **No direct system headers.** Implementation code must not ``#include`` system
  C library headers directly; it goes through the proxy headers in ``hdr/`` so
  the build stays self-contained across overlay and full-build modes. A raw
  ``#include <...>`` of a libc header in ``src/`` is a defect.
* **Macros.** Configuration knobs use the ``LIBC_COPT_`` prefix; internal
  feature/property macros come from ``src/__support/macros`` (e.g.
  ``LIBC_TARGET_ARCH_IS_*``, ``LIBC_TYPES_HAS_*``) rather than ad-hoc ``#ifdef``
  on raw compiler predefines.

ABI and Symbol Visibility
=========================

Namespacing (covered above) controls *internal* boundaries; ABI is a separate
question about the *external* symbol surface, and reviewers should treat it as
its own axis. When a patch adds or changes a public symbol, check:

* **Intentional surface.** Is the new symbol meant to be public?
* **Visibility.** Internal implementation symbols stay hidden. Confirm internal
  helpers are not given external linkage.

Standards Conformance and Undefined Behavior
============================================

This is where libc review differs from a typical LLVM subproject, and where
a reviewer's domain attention matters most.

* **Cite the standard.** A behavioral change should be justified against the
  relevant C or POSIX clause. Reviewers should ask which standard, and which
  version (C17 vs. C23, POSIX issue), governs the behavior and they frequently
  diverge, and "the standard says so" is not reviewable without the citation.
* **Distinguish what the standard actually mandates.** Not every observable
  behavior is required by the standard. Behavior that is *implementation-defined*
  or *unspecified* must be handled deliberately, following the priority order and
  guidelines in :ref:`undefined_behavior`: correct answer > correct answer in the
  wrong format > no answer > crashing >> an incorrect answer; match a known-good
  implementation where it aids differential testing; keep undefined-input
  handling simple; stay self-consistent across platforms and sibling functions;
  and **write the decision down**. A patch that pins down implementation-defined
  or unspecified behavior should add the corresponding entry to
  :ref:`undefined_behavior` so users (and future reviewers) can find it. Do not
  let a test silently lock in incidental behavior that the standard never
  promised.
* **errno mechanics and boundary.** Code that sets ``errno`` from libc runtime
  code uses the ``libc_errno`` macro from ``src/__support/libc_errno.h``, not a
  direct assignment to ``errno`` -- this keeps unit tests from perturbing the
  test process's own ``errno`` and keeps overlay mode pointed at the right
  ``errno`` storage. ``errno`` is set just before returning from the *public*
  entrypoint, not from inside helper functions; helpers should report failure
  with ordinary C++ constructs (e.g. ``cpp::optional``, an error-code return)
  and let the entrypoint translate that into ``errno`` once. POSIX functions
  that return an error number directly (e.g. the pthread family) generally do
  *not* also set ``errno`` so check the family's convention rather than
  assuming one or the other.

Testing
=======

A behavioral patch without a test should not be approved. libc has specific
testing facilities and a reviewer should check the patch uses the right ones:

* **Hermetic vs. unit tests.** libc distinguishes unit tests from hermetic tests
  that link against libc itself. New entrypoints generally need coverage that
  exercises the actual built symbol, not just host libc behavior.
* **errno + return assertions.** Use the provided matchers (e.g. the
  ``ErrnoSetterMatcher`` helpers) to assert on the combination of return value
  and ``errno`` rather than checking them ad hoc.
* **Edge cases and the standard's boundaries.** Tests should cover the explicit
  boundary cases the standard calls out, including any implementation-defined
  behavior the patch documents.
* **Differential / fuzz testing.** For functions where correctness is best
  established against an existing implementation, a fuzzer under ``fuzzing/`` that
  cross-checks against a known-good libc (the approach used for, e.g., the string
  conversion functions) is the expected form of evidence. Note that this couples
  the chosen behavior to the reference implementation; if the patch *intentionally*
  diverges from the reference (e.g. to follow the standard where the reference has
  a known bug), the test must encode the intended behavior, and the divergence
  should be recorded per :ref:`undefined_behavior`.
* **Math functions.** Correctly-rounded math is tested against MPFR (and, for
  some functions, the CORE-MATH library); libc aims to be correctly rounded
  for all IEEE rounding modes, and results must be consistent across
  platforms. A new math entrypoint without MPFR-based accuracy tests is
  incomplete.

Portability, Footprint, and Constraints
=======================================

LLVM-libc targets hosted Linux, GPU (AMDGPU/NVPTX), baremetal embedded, and UEFI
from one source base. Reviewers should weigh:

* **No unexpected dependencies.** Watch for hidden dependencies on dynamic
  allocation, global constructors, the host libc, or platform facilities that do
  not exist on baremetal or GPU targets.
* **Code size.** libc is used on size-constrained embedded targets. Per
  :ref:`undefined_behavior`, large amounts of code to handle unreasonable or
  undefined inputs are discouraged; prefer simple, predictable handling over
  elaborate correction logic.
* **Target guards.** Architecture- or feature-specific code (notably in the
  memory functions) must be guarded with the ``__support/macros`` properties and
  must degrade correctly on targets without the feature.

Performance Changes
===================

Performance claims must be backed by numbers, not asserted. For changes that
claim a speedup:

* ask for benchmark results and the methodology used to produce them
  (build configuration, hardware, workload);
* confirm the change does not regress code size or other targets to win on one;
* check that hot internal helpers are ``LIBC_INLINE`` as discussed above.

Giving Feedback
===============

The general LLVM review etiquette applies. A few libc-specific habits help:

* When asking for a missing build surface or config update, name the specific
  file(s) (Bazel overlay path, the relevant ``entrypoints.txt``) so the author
  can fix them all at once rather than across several review rounds.
* When requesting a behavior decision, point to the relevant standard clause and
  to :ref:`undefined_behavior`, so the contributor knows both the rule and where
  to record the decision.

Approving
=========

Final approval follows the LLVM-wide policy (a reviewer with the appropriate
authority, reasonable confidence of community consensus, ``LGTM``). In addition,
before approving a libc patch, a reviewer should be satisfied that:

* the change is in scope and tested;
* the entrypoint, build, and configuration surfaces are all updated and CI is
  green, including the Bazel build;
* any implementation-defined or unspecified behavior the patch introduces is
  handled per :ref:`undefined_behavior` and documented;
* style, and other conventions are followed.

Finding the Right Reviewer
===========================

LLVM-libc has per-area maintainers listed in
`libc/Maintainers.md <https://github.com/llvm/llvm-project/blob/main/libc/Maintainers.md>`__
For a patch touching one of these areas, prefer pulling in the
listed maintainer rather than relying on the general
``libc`` GitHub label alone.

See Also
========

* `LLVM Code-Review Policy <https://llvm.org/docs/CodeReview.html>`__
* `LLVM Developer Policy <https://llvm.org/docs/DeveloperPolicy.html>`__
* `libc/Maintainers.md <https://github.com/llvm/llvm-project/blob/main/libc/Maintainers.md>`__
* :ref:`contributing`
* :ref:`code_style`
* :ref:`implementation_standard`
* :ref:`entrypoints`
* :ref:`source_tree_layout`
* :ref:`undefined_behavior`
* `How to add a new math function <https://github.com/llvm/llvm-project/blob/main/libc/src/math/docs/add_math_function.md>`__