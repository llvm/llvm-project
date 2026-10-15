# Transforming a Project

`clang-ssaf-driver` runs a source transformation over a whole project as it
builds, and presents what it found: the source edits as a patch, and the
transformation report as diagnostics and as SARIF. It drives projects built
with make through `clang-ssaf-orchestrator-make`, whose requirements it
shares; see {doc}`MakeProjects`.

## Usage

Run it from the project root, with its options before the build command:

```console
$ clang-ssaf-driver --transformation=cpp-bounded-buffers -- make -j8
```

The driver knows which summaries and which whole-program analysis each
transformation needs:

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Transformation
     - What it does
   * - ``cpp-bounded-buffers``
     - Rewrites the raw pointers and arrays that need bounds into
       ``bounded_ptr`` and ``bounded_array``.
```

## Results

A run writes two files into the output directory:

- `<transformation>.patch`, the source edits as one unified diff whose paths
  are relative to the project root, for `patch -p1` from there;
- `<transformation>.sarif`, the transformation report: the findings of every
  translation unit, and every conflict between source edits.

Both are named on stdout, after the orchestrator's own results:

```text
patch: /work/proj/ssaf-output/cpp-bounded-buffers.patch
report: /work/proj/ssaf-output/cpp-bounded-buffers.sarif
```

The report is also printed on stderr, one line per finding the way a compiler
prints a diagnostic, and closes with a count for each rule:

```text
src/shared.h:1:8: note: declarator of a multi-declarator group is not yet rewritten [cpp-bounded-buffers-skipped]
src/shared.h:3:1: error: 2 overlapping replacement(s) at file:///work/proj/src/shared.h byte 20 were dropped; resolve manually. [clang-reforge-replacement-conflict]
src/shared.h:5:1: note: candidate edit: "bounded_ptr<int> "
src/shared.h:3:1: note: candidate edit: "bounded_ptr<long> "
1 clang-reforge-replacement-conflict, 1 cpp-bounded-buffers-skipped reported
```

A finding is a place the transformation did not rewrite. A conflict is a place
where translation units asked for different edits; none of them is made, and
an edit that overlaps a conflict is left out of the patch too, with a warning.
Both are left to be done by hand, and neither changes the exit status.

## Options

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Option
     - Purpose
   * - ``--transformation=<name>``
     - The transformation to run. Required.
   * - ``-o``, ``--output-dir=<dir>``
     - Where the results and the SSAF artifacts go. Defaults to
       ``ssaf-output``, and is the orchestrator's output directory.
   * - ``--ssaf-clang=<path>``, ``--ssaf-linker=<path>``,
       ``--ssaf-analyzer=<path>``, ``--ssaf-src-edit-merge=<path>``,
       ``--ssaf-report-merge=<path>``,
       ``--ssaf-conflict-merge=<path>``,
       ``--ssaf-extra-clang-arg=<arg>``, ``-k``, ``-v``
     - Passed to the orchestrator; see {doc}`MakeProjects`.
   * - ``--version``
     - The clang version this was built from.
```

## What is not supported

**Edits outside the project root.** A file the project does not contain is
left out of the patch and is not applied to; the driver names it in a warning.

**Compilations that run in another directory**, as in recursive make or
`make -C`. A source edit names its file relative to where its compilation ran,
and is merged where the link runs, so such an edit can be lost or land on the
wrong file. Run the build from the project root.
