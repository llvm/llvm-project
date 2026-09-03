# Clang SYCL Linker

```{contents}
:local:
```

## Introduction

This tool works as a wrapper around the SYCL device code linking process.
The purpose of this tool is to provide an interface to link SYCL device bitcode
in LLVM IR format together with LLVM bitcode libraries, and then run SPIR-V
code generation on the fully linked device code to produce the final output.
After the linking stage, the fully linked device code in LLVM IR format may
undergo several SYCL-specific finalization steps before the SPIR-V code
generation step.
The tool will also support the Ahead-Of-Time (AOT) compilation flow. AOT
compilation is the process of invoking the back-end at compile time to produce
the final binary, as opposed to just-in-time (JIT) compilation when final code
generation is deferred until application runtime.

Device code linking for SYCL offloading has known quirks that make it
difficult to use in a unified offloading setting. The primary issue is that
several finalization steps are required to be run on the fully linked LLVM
IR bitcode to guarantee conformance to SYCL standards. This step is unique to
the SYCL offloading compilation flow.

This tool has been proposed to work around this issue.

## Usage

This tool can be used with the following options. Several of these options will
be passed down to downstream AOT compilation tools like 'ocloc' and 'opencl-aot'.

```console
OVERVIEW: A utility that wraps around the SYCL device code linking process.
This enables LLVM IR linking, post-linking and code generation for SPIR-V
JIT and AOT targets.

USAGE: clang-sycl-linker [options] <input bitcode files>

OPTIONS:
  --arch <value>                Specify the name of the target architecture.
  --dry-run                     Print generated commands without running.
  -help-hidden                  Display all available options
  -help                         Display available options (--help-hidden for more)
  -L <dir>                      Add <dir> to the library search path
  -l <libname>                  Search for library <libname>
  --whole-archive               Include all archive members in the link
  --no-whole-archive            Only include archive members that resolve undefined symbols (default)
  -u <symbol>                   Force undefined symbol during linking
  --module-split-mode=<mode>    Module split mode: 'translation_unit' (default), 'kernel', or 'link_unit'
  --ocloc-options=<value>       Options passed to ocloc for Intel GPU AOT compilation
  --opencl-aot-options=<value>  Options passed to opencl-aot for Intel CPU AOT compilation
  -o <path>                     Path to file to write output
  --save-temps                  Save intermediate results
  --triple <value>              Specify the target triple.
  --version                     Display the version number and exit
  -v                            Print verbose information
  -spirv-dump-device-code=<dir> Directory to dump SPIR-V IR code into
```

## Library Linking

Device bitcode libraries can be packaged into archive libraries (`.a` files)
using `llvm-ar` and linked using the `-l` option:

```console
llvm-ar rc libdevice.a func1.bc func2.bc func3.bc
clang-sycl-linker input.bc -l device -L /path/to/libs
```

The linker supports standard archive library search semantics:

- `-l <name>` searches for `lib<name>.a` in the directories specified by `-L`
- `-l :<exact-name>` searches for the exact filename in the `-L` paths
- Absolute paths can be passed as positional arguments: `clang-sycl-linker input.bc /path/to/libdevice.a`

By default, archive linking is **lazy** - only archive members (individual `.bc` files)
that resolve undefined symbols are extracted and linked. This happens at file
granularity: if any symbol in a `.bc` file is needed, all symbols in that file
are included. The linker uses a symbol-driven fixed-point algorithm: it
repeatedly scans archives to extract members that resolve currently undefined
symbols until no more extractions occur.

To force extraction of all archive members regardless of symbol resolution, use
`--whole-archive`:

```console
clang-sycl-linker input.bc --whole-archive -l device --no-whole-archive -l other
```

The `-u <symbol>` option can be used to force a symbol to be undefined, which
can trigger extraction of archive members that define that symbol:

```console
clang-sycl-linker input.bc -u my_init_function -l device
```

## Examples

### Basic Usage

This tool is intended to be invoked when targeting any of the target offloading
toolchains. When the --sycl-link option is passed to the clang driver, the
driver will invoke the linking job of the target offloading toolchain, which in
turn will invoke this tool. This tool can be used to create one or more fully
linked device images that are ready to be wrapped and linked with host code to
generate the final executable.

```console
clang-sycl-linker --triple spirv64 --arch bmg_g21 input.bc
```

### Linking with Device Libraries

To link device bitcode libraries, first package them into archive files:

```console
# Create device library archives
llvm-ar rc libmath.a sin.bc cos.bc tan.bc
llvm-ar rc libutils.a helper1.bc helper2.bc

# Link with lazy loading (only needed members extracted)
clang-sycl-linker --triple spirv64 kernel.bc -l math -l utils -L /path/to/libs -o kernel.spv

# Force all members to be included from libmath.a
clang-sycl-linker --triple spirv64 kernel.bc --whole-archive -l math --no-whole-archive -l utils -L /path/to/libs -o kernel.spv

# Use exact archive filename or absolute path
clang-sycl-linker --triple spirv64 kernel.bc -l :libmath.a -L /path/to/libs -o kernel.spv
clang-sycl-linker --triple spirv64 kernel.bc /absolute/path/libmath.a -o kernel.spv
```
