# Make Orchestrator Shims

`clang-ssaf-orchestrator-make` analyses a project by riding along with its
build. It does that with a set of small programs — *shims* — that stand in
for the tools the build invokes. This describes how they are selected, what
each one does, and how they learn what run they are part of; the user-facing
behaviour is in {doc}`../user-docs/MakeProjects`.

## Setting up the build's environment

At the start of a run the orchestrator does two things to the environment the
build will see.

It fills `<output-dir>/shims/` with links named after the build tools, all
pointing at one shim program, and prepends that directory to `PATH`. A build
can spell the compiler several ways, so several names lead to the same shim:

```text
cc, clang, c++, clang++  ->  clang
ld                       ->  ld
ar                       ->  ar
```

It also writes the configuration for the shims into a JSON document and puts
that document's path in `SSAF_CONFIG`. A shim that finds no `SSAF_CONFIG`
reports an error and exits without running anything.

## Running the binary tool

A shim has to run the tool the build asked for, by the name the build spelled,
but on the modified `PATH` that name finds the shim itself.
`SSAFWorkflowConfig.restored_path_env_var` therefore holds the build's `PATH`
as it was before the shim link directory went in front of it.
`Shim.bin_tool_path` looks the tool up on that `PATH`, and
`Shim.get_unmodified_env` gives the tool an environment with that `PATH`
restored and `SSAF_CONFIG` removed.

## The shim program

`libexec/ssaf-orchestrator-shims/` holds one program, `shim`, and every link
points at it. It puts the package on `sys.path`, looks up the name it was
invoked under in `SHIM_FOR_LINK_NAME`, the table the links were made from, and
imports the module that stands in for that tool. It is a file rather than a
package module because it is symlinked under several names, and `argv[0]` is
what tells it which tool it stands in for.

The shim bodies live in the package, as `clang_ssaf_orchestrator_make.shims`,
split by what they do:

```text
shim.py          the Shim class, its diagnostics, merge_edits
commands.py      Files, is_object, is_archive, artifact_of, summary_of,
                 filter_build_artifacts, unique
driver_jobs.py   Job and the -### listing reader
run.py           run_shim
clang.py ld.py ar.py
```

Every shim has the same shape, which `run_shim` implements: run the tool the
build asked for, then do the SSAF work. A failure in the first half is the
build's own. A failure in the second is logged, and the build carries on.

## The clang shim has two roles

One program serves both, and a marker on the command line tells them apart.

**Invoked as the clang driver.** The build spelled `cc`, `clang`, `c++` or
`clang++`, and `PATH` found the link. The shim runs the binary tool on the
build's command line untouched, asks it with `-###` how that command
decomposes into jobs, and forwards each job to the shim that stands in for its
tool.

**Invoked as cc1 by the shim's driver role.** A forwarded job ends in
`--ssaf-shim-skip-binary-tool-invocation`, which only the driver role appends.
The binary tool has therefore already run, and the shim does only the SSAF
work: it runs the SSAF clang over the job with the `--ssaf-*` options added.
`Shim.__init__` removes the marker as the command enters the shim, so nothing
downstream sees it.

## Matching a job to a shim

For each job in the driver's `-###` listing, take the program the job names,
strip its directory and any trailing `-<digits>`, and look the result up among
the links:

```text
/…/toolchains/…/usr/bin/clang   ->  clang     ->  <output-dir>/shims/clang
/…/build/bin/clang-<N>          ->  clang     ->  <output-dir>/shims/clang
/usr/bin/ld                     ->  ld        ->  <output-dir>/shims/ld
/usr/bin/dsymutil               ->  dsymutil  ->  no link; nothing stands in
```

The suffix is stripped because a driver names its own binary `clang` when
installed and `clang-<N>`, after its major version, in a build tree. A job
with no matching link is noted and skipped: the build's compiler has already
run it, and there is no SSAF step for its tool.

## Recording what the build assembled

`ld` and `ar` each read their command line for the artifact it produces and
the artifacts it consumes, and record them in
`<output-dir>/build-graph/<path>.buildstep.json`. The top-most link unit is
found from these records, and they tell an artifact the build assembled from
one it never touched.

Everything a shim writes is named for a build artifact's path plus a suffix,
such as `ssaf-artifacts/<path>.summary.json`.

## Failure logs

A failure log is a diagnostic about a command rather than something the
command produced, so it lives apart, under `failure-logs/`. The run counts
every log at its end, which is how a failure a shim logged and moved past
still fails the run.

A log is named for the artifact it concerns: the output of the command that
failed, or the input whose summary is missing. A command that names no output,
such as `clang -c a.c b.c`, and an internal error in a shim are logged instead
under the directory the command ran in, named for a hash of the command:

```text
failure-logs/<path>.<kind>.log        named for an artifact
failure-logs/<dir>/<hash>.<kind>.log  named for a command
```

The kind says what went wrong; the kinds are the values of `SSAFFailureKind`:

```text
extract      summary extraction for a compilation
transform    source-edit generation for a compilation
link         clang-ssaf-linker over a link's inputs
archive      the same for an archive
merge        clang-ssaf-src-edit-merge over a link's or an archive's inputs
forward      handing a job to the shim that stands in for its tool
missing      an artifact with nothing analysed behind it
driver-jobs  the driver refused to list a command's jobs
internal     an unhandled exception inside a shim
```

## Carrying the link unit name into the second pass

One piece of state flows from the first pass to the second: the top-most link
unit's own name, which is not an artifact id and is unknown until the first
pass's linker has written the link unit summary.

The orchestrator reads it from that summary, as the `lu_namespace` component
whose kind is `LinkUnit`, and puts it in the transformation pass's
configuration; the clang shim passes it as `--ssaf-link-unit-id=` to every
source-edit generation. This is why the second pass's configuration file
cannot be written up front.
