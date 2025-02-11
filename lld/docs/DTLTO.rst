Distributed ThinLTO (DTLTO)
===========================

DTLTO allows for the distribution of backend ThinLTO compilations via external
distribution systems, e.g. Incredibuild. There is existing support for
distributing ThinLTO compilations by using separate thin-link, backend
compilation, and link steps coordinated by a build system that can handle the
dynamic dependencies specified by the index files, such as Bazel. However, this
often requires changes to the user's build process. DTLTO distribution is
managed internally in LLD as part of the traditional link step and, therefore,
should be usable via any build process that can support in-process ThinLTO.

ELF LLD
-------

The command line interface for DTLTO is:

- `--thinlto-distributor=<path>`
  Specifies the file to execute as a distributor process.
  If specified, ThinLTO backend compilations will be distributed.

- `--thinlto-remote-opt-tool=<path>`
  Specifies the path to the tool that the distributor process will use for
  backend compilations.

  The remote optimisation tool invoked must match the version of LLD.

  Currently `Clang` is used on remote machines to perform optimization. The
  design permits this to be swapped out later without affecting distributors.
  This may occur in the future, at which point a different set of constraints
  will apply.

- `-mllvm -thinlto-distributor-arg=<arg>`  
  Specifies `<arg>` on the command line when invoking the distributor.  

- `-mllvm -thinlto-remote-opt-tool-arg=<arg>`  
  Specifies `<arg>` on the command line to the remote optimisation tool. These
  arguments are appended to the end of the command line for the remote 
  optimisation tool.

Remote optimisation tool options that imply an additional input or output file 
dependency are unsupported and may result in miscompilation depending on the
properties of the distribution system (as such additional input/output files may
not be pushed to or fetched from distribution system nodes correctly). If such 
options are required, then the distributor can be modified to accept switches 
that specify additional input/output dependencies, and 
`-Xdist`/`-thinlto-distributor-arg=` can be used to pass such options through 
to the distributor.

Some LLD LTO options (e.g., `--lto-sample-profile=<file>`) are supported. 
Currently, other options are silently accepted but do not have the desired 
effect. Support for such options will be expanded in the future.

COFF LLD
--------

The command line interface for COFF LLD is generally the same as for ELF LLD.

Currently, there is no DTLTO command line interface supplied for `Clang-cl`, as
users are expected to invoke LLD directly.
