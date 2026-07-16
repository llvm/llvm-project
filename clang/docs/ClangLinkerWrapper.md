# Clang Linker Wrapper

```{contents}
:local: true
```

(clang-linker-wrapper-1)=

## Introduction

This tool works as a wrapper of the normal host linking job. This tool is used
to create linked device images for offloading and the necessary runtime calls to
register them. It works by first scanning the linker's input for embedded device
offloading data stored at the `.llvm.offloading` section. This section
contains binary data created by the `llvm-offload-binary` utility. The
extracted device files will then be linked. The linked modules will then be
wrapped into a new object file containing the code necessary to register it with
the offloading runtime.

## Usage

This tool can be used with the following options. Any arguments not intended
only for the linker wrapper will be forwarded to the wrapped linker job.

```console
USAGE: clang-linker-wrapper [options] -- <options to pass to the linker>

OPTIONS:
  --cuda-path=<dir>      Set the system CUDA path
  --device-debug         Use debugging
  --device-linker=<value> or <triple>=<value>
                         Arguments to pass to the device linker invocation
  --dry-run              Print program arguments without running
  --help-hidden          Display all available options
  --help                 Display available options (--help-hidden for more)
  --host-triple=<triple> Triple to use for the host compilation
  --linker-path=<path>   The linker executable to invoke
  -L <dir>               Add <dir> to the library search path
  -l <libname>           Search for library <libname>
  --opt-level=<O0, O1, O2, or O3>
                         Optimization level for LTO
  --override-image=<kind=file>
                          Uses the provided file as if it were the output of the device link step
  -o <path>              Path to file to write output
  --pass-remarks-analysis=<value>
                         Pass remarks for LTO
  --pass-remarks-missed=<value>
                         Pass remarks for LTO
  --pass-remarks=<value> Pass remarks for LTO
  --print-wrapped-module Print the wrapped module's IR for testing
  --ptxas-arg=<value>    Argument to pass to the 'ptxas' invocation
  --relocatable           Link device code to create a relocatable offloading application
  --save-temps           Save intermediate results
  --sysroot<value>       Set the system root
  --verbose              Verbose output from tools
  -v
  --wrapper-verbose      Verbose output from the linker-wrapper
  --version              Display the version number and exit
  --                     The separator for the wrapped linker arguments
```

The linker wrapper will generate the appropriate runtime calls to register the
generated device binary with the offloading runtime. To do this step manually we
provide the `llvm-offload-wrapper` utility.

## Relocatable Linking

The `clang-linker-wrapper` handles linking embedded device code and then
registering it with the appropriate runtime. Normally, this is only done when
the executable is created so other files containing device code can be linked
together. This can be somewhat problematic for users who wish to ship static
libraries that contain offloading code to users without a compatible offloading
toolchain.

When using a relocatable link with `-r`, the `clang-linker-wrapper` will
perform the device linking and registration eagerly. This will remove the
embedded device code and register it correctly with the runtime. Semantically,
this is similar to creating a shared library object. If standard relocatable
linking is desired, simply do not run the binaries through the
`clang-linker-wrapper`. This will simply append the embedded device code so
that it can be linked later.

## Matching

The linker wrapper will link extracted device code that is compatible with each
other. Generally, this requires that the target triple and architecture match.
An exception is made when the architecture is listed as `generic`, which will
cause it be linked with any other device code with the same target triple.

## Debugging

The linker wrapper performs a lot of steps internally, such as input matching,
symbol resolution, and image registration. This makes it difficult to debug in
some scenarios. The behavior of the linker-wrapper is controlled mostly through
metadata, described in [clang documentation](https://clang.llvm.org/docs/OffloadingDesign.html).

The individual tool invocations the wrapper performs can be printed with the
`--wrapper-verbose` flag, and the intermediate files they operate on can be
kept with `--save-temps`. When both are enabled the wrapper emits a
self-contained sequence of commands that reproduce its output. The example below
shows the sequence for a single OpenMP image.

```sh
$> clang openmp.c -fopenmp --offload-arch=gfx90a -c
$> clang openmp.o -fopenmp --offload-arch=gfx90a -Wl,--wrapper-verbose -Wl,--save-temps

# 1. Extract each embedded device image from the host object.
llvm-offload-binary openmp.o --image=kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a,file=openmp.gfx90a.o

# 2. Link the extracted image for the device target.
clang --target=amdgcn-amd-amdhsa -mcpu=gfx90a openmp.gfx90a.o -o openmp.gfx90a.img <...>

# 3. Bundle the linked image back into the offloading binary format.
llvm-offload-binary -o openmp.gfx90a.offload --image=file=openmp.gfx90a.img,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a

# 4. Generate the host runtime registration code for the bundled images.
llvm-offload-wrapper --kind=openmp --triple=x86_64-unknown-linux-gnu -o openmp.wrapper.bc openmp.gfx90a.offload

# 5. Compile the registration code into a host object.
clang --target=x86_64-unknown-linux-gnu -c -fPIC -o openmp.wrapper.o openmp.wrapper.bc

# 6. Link the host objects with the registration code into the executable.
ld.lld openmp.host.o openmp.wrapper.o -o a.out <...>
```

To replace the output of a single stage, edit the relevant intermediate file and
re-run the remaining commands. To bypass the device link entirely and substitute
a pre-built image, use the `--override-image=<kind>=<file>` flag.

## Example

This tool links object files with offloading images embedded within it using the
`-fembed-offload-object` flag in Clang. Given an input file containing the
magic section we can pass it to this tool to extract the data contained at that
section and run a device linking job on it.

```console
clang-linker-wrapper --host-triple=x86_64 --linker-path=/usr/bin/ld -- <Args>
```
