# Make-Based Projects

Running SSAF over a whole project means running summary extraction for every
translation unit the project compiles, and linking those summaries the way the
build links the objects they describe. `clang-ssaf-orchestrator-make` does that
for projects built with make. The design based on tool shims is documented in
{doc}`../developer-docs/MakeOrchestratorShims`.

The build is executed normally, and each step it takes is paired with an SSAF
step:

- A **compilation** gets a second compilation — the frontend job the build's
  own compiler said it would run, with the `--ssaf-*` options appended and
  nothing taken away, so extraction sees exactly what the build compiles.
- An **archive** and a **link** get a `clang-ssaf-linker`
  invocation over the summaries of what went into them. These are not copies
  of the build's command and carry no `--ssaf-*` options.

The mapping onto the SSAF tools is direct:

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Build step
     - SSAF step
   * - ``$(CXX) -c -o parser.o parser.cpp``
     - ``clang -cc1 … -o parser.o -x c++ parser.cpp
       --ssaf-extract-summaries=…``
   * - ``$(AR) rcs libparse.a parser.o lexer.o``
     - ``clang-ssaf-linker static-library create parser.json lexer.json``
   * - ``$(CXX) -o app main.o libparse.a``
     - ``clang-ssaf-linker main.json libparse.json``
```

## Usage

Run it from the project root; that is where the build runs.

There are three workflows, each going one step further than the last:

- **Summary extraction and linking**, with `--extract-summaries`. Every
  translation unit gets a summary, and the summaries are linked the way the
  build links the objects, into one link unit summary.
- **Whole-program analysis**, adding `--analysis`. `clang-ssaf-analyzer` runs
  the named analyses over the link unit summary.
- **Whole-program transformation**, adding `--transformation`. A second build
  generates source edits from the analysis result, and they are merged across
  the whole program.

Summary extraction and linking:

```console
$ clang-ssaf-orchestrator-make \
    --extract-summaries=PointerFlow,UnsafeBufferUsage \
    -o ssaf-output \
    -- make -j8
```

Whole-program analysis:

```console
$ clang-ssaf-orchestrator-make \
    --extract-summaries=PointerFlow,UnsafeBufferUsage \
    --analysis=UnsafeBufferReachableAnalysisResult \
    -o ssaf-output \
    -- make -j8
```

Whole-program transformation, which ends in merged source edits:

```console
$ clang-ssaf-orchestrator-make \
    --extract-summaries=PointerFlow,UnsafeBufferUsage \
    --analysis=UnsafeBufferReachableAnalysisResult \
    --transformation=cpp-bounded-buffers \
    -o ssaf-output \
    -- make -j8
```

For a project configured by autoconf, run `configure` first and give the
orchestrator only the `make` invocation.

## Options

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Option
     - Purpose
   * - ``--extract-summaries=<name,...>``
     - Summary extractors to run over every translation unit. Required.
   * - ``--analysis=<name>``
     - Whole-program analysis to run on the link unit summary. Repeatable.
       Without it the run stops after summary extraction.
   * - ``--transformation=<name>``
     - Source transformation to run in the second pass. Requires
       ``--analysis``. Without it the run stops after the analysis.
   * - ``-o``, ``--output-dir=<dir>``
     - Where the SSAF artifacts go, made absolute. Defaults to
       ``ssaf-output``. It must be absent, empty, or an earlier run's.
   * - ``--ssaf-clang=<path>``
     - SSAF-capable clang used for the SSAF compilations. Defaults to
       ``clang`` on ``PATH``.
   * - ``--ssaf-linker=<path>``
     - Path to ``clang-ssaf-linker``. Defaults to the name on ``PATH``.
   * - ``--ssaf-analyzer=<path>``
     - Path to ``clang-ssaf-analyzer``. Defaults to the name on ``PATH``, and
       is only required with ``--analysis``.
   * - ``--ssaf-src-edit-merge=<path>``
     - Path to ``clang-ssaf-src-edit-merge``. Defaults to the name on
       ``PATH``, and is only required with ``--transformation``.
   * - ``--ssaf-extra-clang-arg=<arg>``
     - Extra argument for every SSAF compilation. Repeatable.
   * - ``-k``, ``--keep-going``
     - Exit successfully even though some SSAF steps failed. The failures are
       still reported.
   * - ``-v``, ``--verbose``
     - Print every command the orchestrator and the shims run, on stderr.
   * - ``--version``
     - The clang version this was built from.
```

Each `--ssaf-*` tool must resolve to one executable file. A value naming more
than one word, a directory, something that is not executable, or a path inside
our own installed shim directory — which is what an earlier run's links on
`PATH` look like — is refused, and every unresolvable tool is reported at
once.

## Artifacts

A build artifact is named by its resolved **path**, so two spellings of one
file name the same artifact. That path is what a buildstep record names, and
the value of `--ssaf-compilation-unit-id=`.

An SSAF artifact is filed under the same path with its leading separator
dropped, so an object at `/work/parse/build/parser.o` gets a summary at
`<output-dir>/ssaf-artifacts/work/parse/build/parser.o.summary.json`.

No part of the path is elided, because the path is what tells two
translation units apart. A build that compiles one source into
`build/first/parser.o` and another into `build/second/parser.o` has two
translation units, and only the whole path tells them apart.

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Path
     - Contents
   * - ``<output-dir>/ssaf-artifacts/<path>.summary.json``
     - What SSAF holds for ``<path>``: a TU summary where the path names a
       compilation's object, and whatever the SSAF linker wrote where the
       build assembled it.
   * - ``<output-dir>/build-graph/<path>.buildstep.json``
     - Which build artifacts were inputs for the ``<path>`` build artifact.
   * - ``<output-dir>/wpa.json``
     - Whole-program analysis result.
   * - ``<output-dir>/ssaf-artifacts/<path>.edits.yaml``
     - Source edits for the ``<path>`` build artifact.
   * - ``<output-dir>/ssaf-artifacts/<path>.conflicts.sarif``
     - Edits dropped because they conflicted.
   * - ``<output-dir>/ssaf-artifacts/<path>.report.sarif``
     - Transformation report for the ``<path>`` build artifact.
   * - ``<output-dir>/config-analysis.json``
     - Configuration for the analysis pass, the first build pass.
   * - ``<output-dir>/config-transformation.json``
     - Configuration for the transformation pass, the second build pass.
   * - ``<output-dir>/shims/``
     - Links to the shims named so they interpose the binary tools, recreated
       on every run.
   * - ``<output-dir>/.clang-ssaf-orchestrator-make``
     - SSAF ownership attribute for cleanup.
   * - ``<output-dir>/failure-logs/<path>.<kind>.log``
     - What went wrong for ``<path>``: a command writing it whose SSAF step
       failed, or an input that had no summary.
   * - ``<output-dir>/failure-logs/<dir>/<hash>.<kind>.log``
     - What went wrong for a command that names no output, and any internal
       error in a shim; ``<hash>`` names the command and ``<dir>`` is where
       it ran. Every ``.log`` anywhere under the output directory counts
       against the run; a log is always a failure.
```

The buildstep records are what the top-most link unit is found from. An
input contributes the summary filed under its own id or nothing at all,
so an artifact whose SSAF step failed takes the artifacts behind it down
with it, and the run says so and exits non-zero.

## Failure handling

A failing build step fails the run with the build's own exit status, and no
SSAF step runs for it.

A failing SSAF step never fails the build: it is reported and logged to a
`.log` under `<output-dir>/failure-logs/`, and the build carries on. The run
exits non-zero afterwards, because the artifacts are incomplete. `-k` makes the
run exit zero instead; the failures are still reported, so the message appears
either way.

## Interposition

The orchestrator populates `<output-dir>/shims/` with links named after the
tools a build invokes — `cc`, `clang`, `c++`, `clang++`, `ld`, `ld.lld` and
`ar` — each pointing at the shim that stands in for it, and prepends that
directory to `PATH`. It also puts the path of a configuration file in
`SSAF_CONFIG`.

It changes nothing else in the build: no variable is replaced, so whatever the
Makefile put in `CXX` — flags, a dialect, a define — arrives at the compiler
exactly as it was written, and at the SSAF compilation with it.

## What is not supported

**Compilers other than clang.** Every compilation gets an SSAF step replaying
the frontend job the build's own compiler said it would run, and that replay
carries the `--ssaf-*` options, which only clang has. A project built by GCC
therefore cannot be analysed, and the orchestrator does not pretend otherwise:
`gcc` and `g++` are not among the names it interposes, so such a build is left
alone rather than analysed into an empty result. On a platform where `gcc` and
`g++` are clang this means a build spelling those names is not analysed
either. GNU make defaults `CXX` to `g++`,
so a Makefile that never sets it needs `CXX=clang++` on the make command line.
The clang the build uses and the one named by `--ssaf-clang` must also be
compatible with each other, since the frontend command line one produces is
replayed by the other.

**Compiler launchers**, such as `ccache`, `distcc`, `sccache` and `xcrun`.
Run the build without the launcher instead, by overriding the variable that
adds it on the make command line, as in `-- make CC=clang CXX=clang++`.

**A compiler named by an absolute path**, in a recipe or through a variable:

```make
CXX = /opt/toolchain/bin/clang++     # PATH is never consulted
CC := $(shell xcrun -find clang)     # an absolute path at parse time
```

Name the compiler by its bare name on the make command line instead, with its
directory on `PATH`:

```console
$ PATH=/opt/toolchain/bin:$PATH clang-ssaf-orchestrator-make \
    --extract-summaries=PointerFlow -- make CXX=clang++
```

**Inputs named indirectly on a link line.** A link that outgrows `ARG_MAX` and
passes its objects through ld64's `-filelist`, given to `ld` or to the driver,
is read as it stands, so those objects are not seen and the step contributes
no link unit summary.

**Linkers not named `ld`.** A link the driver hands to `ld.lld`, `ld64.lld`,
`ld.gold` or `mold`, through `-fuse-ld=` or the toolchain's default, has no
shim standing in for it, so it contributes nothing, and a project linked that
way has no link unit.

**Frontend command lines.** A recipe running `clang -cc1 …` directly, or
`clang @file.rsp` whose first token is `-cc1`, is compiled, but its SSAF step
fails: the driver cannot list the jobs of a frontend command line.

**Assembler sources.** A `.s` or `.S` input carries a `-cc1as` job, whose SSAF
step fails.

## Implementation details

How a compilation decomposes is worked out by asking the compiler driver, using
the `-###` option.

Because a compilation's SSAF step replays the build's own action, it writes the
build's outputs a second time — the object, and any dependency file or
serialized diagnostics the job asked for — with the SSAF clang rather than the
build's. Build products are not preserved; this is an analysis run, not a
build.

Every build gets make's `-B` appended, because a translation unit that make
considers up to date is not compiled and so gets no SSAF step. A build command
that is not make draws a warning.

Each run starts by deleting the output directory, recreating it and marking it
with a `.clang-ssaf-orchestrator-make` file. An existing output directory is
deleted only when it is empty or carries that mark, and never when it is, or
contains, the project.
