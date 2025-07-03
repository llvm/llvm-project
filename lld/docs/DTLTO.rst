Integrated Distributed ThinLTO (DTLTO)
======================================

Integrated Distributed ThinLTO (DTLTO) enables the distribution of backend
ThinLTO compilations via external distribution systems, such as Incredibuild,
during the traditional link step.

The implementation is documented here: https://llvm.org/docs/DTLTO.html.

Currently, DTLTO is only supported in ELF LLD. Support will be added to other
LLD flavours in the future.

ELF LLD
-------

The command-line interface is as follows:

- ``--thinlto-distributor=<path>``  
  Specifies the file to execute as the distributor process. If specified,
  ThinLTO backend compilations will be distributed.

- ``--thinlto-remote-compiler=<path>``  
  Specifies the path to the compiler that the distributor process will use for
  backend compilations. The compiler invoked must match the version of LLD.

- ``--thinlto-distributor-arg=<arg>``  
  Specifies ``<arg>`` on the command line when invoking the distributor.
  Can be specified multiple times.

- ``--thinlto-remote-compiler-arg=<arg>``  
  Appends ``<arg>`` to the remote compiler's command line.
  Can be specified multiple times.

  Options that introduce extra input/output files may cause miscompilation if
  the distribution system does not automatically handle pushing/fetching them to
  remote nodes. In such cases, configure the distributor - possibly using
  ``--thinlto-distributor-arg=`` - to manage these dependencies. See the
  distributor documentation for details.

Some LLD LTO options (e.g., ``--lto-sample-profile=<file>``) are supported.
Currently, other options are silently accepted but do not have the intended
effect. Support for such options will be expanded in the future.
