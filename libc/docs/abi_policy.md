(abi_policy)=

# ABI and Public Header Policy

LLVM-libc defines its own ABI. This page explains how the layout of public
types is chosen, which symbols the library exports, and how public headers are
written. It applies to full builds, where LLVM-libc provides the headers. In
{ref}`overlay mode <overlay_mode>` the headers and the ABI belong to the system
libc.

See {ref}`design_philosophy` for why the project works this way.

## Nothing is frozen

LLVM-libc currently makes no ABI stability promise. The layouts of public
types, the names of internal symbols, and the values of implementation-defined
macros may change between two commits. Because the library is
{ref}`linked statically into the program it was built for <design_philosophy>`,
such a change does not break existing binaries.

This lack of stability gives the project the freedom to change the ABI without
fear of breaking users. If there's a mistake in the ABI, it can be fixed quickly
and uncontroversially.

The project intends to keep this freedom for as long as possible. A change that
would make part of the ABI hard to change later needs discussion with the
maintainers before it is written.

(choosing_type_layout)=

## Choosing the layout of a public type

When a public type is added or changed, its layout is decided by the first of
these that applies:

1. **The kernel.** If the type is passed to or received from a system call
   without conversion, it must have the layout the kernel expects. `struct
   timespec` and `stack_t` are examples.
2. **LLVM-libc.** Otherwise, design the type for LLVM-libc's implementation,
   and convert at the system call boundary where necessary. `struct stat` is
   LLVM-libc's own layout and is filled in from `statx`
   (`src/__support/OSUtil/linux/stat/`). `struct sigaction` is converted to and
   from the kernel's layout (`KernelSigaction` in
   `src/signal/linux/signal_utils.h`).
3. **An existing implementation**, only when there is no other source of
   truth. This happens when the kernel's UAPI headers are wrong or unusable
   for user space, or when an interface has no real standard and one
   implementation has become the de facto definition, as glibc has for
   `termios`.

"A program built against another libc's headers expects this layout" is not a
reason to choose that layout. Programs that rely on details a standard does not
promise should be fixed, unless no standard exists at all.

### Opaque types

Where a standard allows a type to be opaque, keep it opaque. As an example,
LLVM-libc's `pthread_t` is a struct, which POSIX explicitly permits. Keeping it
a struct means its contents can change later without breaking correct programs.
Correct code that needs to compare thread IDs uses `pthread_equal`.

`FILE` and `DIR` are forward declarations in the public headers. Nothing
outside the library can depend on their contents.

### Widths of common types

In general it's best to have a single definition of a given type. A type that
differs between platforms requires extra care wherever it is handled. The `long`
type is an example: it is 32 or 64 bits wide depending on the target. `long` is
defined by the platform and the compiler, so LLVM-libc has to handle both
widths and cannot standardize it.

For types that are provided by LLVM-libc, it's generally preferred to use the
larger type. Smaller values will fit in larger types without issue, but the
reverse is not true.

`time_t` is 64 bits on every target, because 32-bit time ends in 2038. The
types that travel with it, such as `off_t`, `dev_t`, and `rlim_t`, are also 64
bits everywhere. This simplifies the build, and it means the `*64` variants of
file functions are not needed.

A few related types (`ino_t`, `nlink_t`, `blkcnt_t`) are still pointer width.
That is unfinished work and not a design decision.

## Feature-test macros

Public headers do not use feature-test macros such as `_GNU_SOURCE` or
`_POSIX_C_SOURCE` to hide declarations. Do not add `#ifdef` guards of that kind
to a header. The exception is a macro that a standard itself defines for this
purpose, such as `__STDC_WANT_LIB_EXT1__` for Annex K.

Headers are generated from the YAML files in `libc/include/` (see
{ref}`header_generation`), and a generated header declares only the functions
that are enabled for the target. A user who wants a different API surface
generates headers to match it, by pointing the `LIBC_CONFIG_PATH` CMake option
at their own `entrypoints.txt`, `headers.txt`, and configuration (see
{ref}`custom_config_directory`), instead of every user receiving one set of
headers full of conditionals.

Each function in the YAML files lists the `standards` it comes from (`stdc`,
`posix`, `gnu`, `bsd`, `linux`, `llvm_libc_ext`, and so on). Keep that field
accurate. It records where each declaration comes from.

## Exported symbols

Every public function is defined in the internal `LIBC_NAMESPACE` namespace and
exported under its standard C name by the `LLVM_LIBC_FUNCTION` macro (see
{ref}`implementation_standard`). The default namespace name includes the LLVM
version, and {ref}`Hand-in-Hand <hand_in_hand>` consumers compile the shared
code under a different namespace, so several copies of LLVM-libc code can be
linked into one program. Nothing may depend on the mangled names of internal
symbols.

### Aliases for another libc's symbols

LLVM-libc does not export another libc's private symbols in general.

The exception is the `LIBC_COPT_PROVIDE_GNU_ALIASES` option, which is enabled
for Linux. It provides a short list of glibc symbol names, such as
`__errno_location`, because some users must link precompiled libraries that
were built against glibc. The list is kept short deliberately. It is a
compromise for those users and not a step toward binary compatibility with
glibc. Additions need agreement from the maintainers.

Symbols that a compiler or a C++ standard library requires of any libc, such
as `__stack_chk_fail` or `__cxa_thread_atexit_impl`, are ordinary entrypoints
and are not covered by this restriction.

### Customization points

Some macros exist so that downstream users can control how LLVM-libc code is
compiled into their own library. Downstream users depend on `LIBC_INLINE` and
`LIBC_INLINE_VAR` being used on every inline function and variable, and
`LLVM_LIBC_FUNCTION_ATTR` lets a packager attach attributes to every public
function. Use these macros as documented in {ref}`code_style`, and do not
remove or bypass them.

## Kernel headers

A full build for Linux requires the kernel's UAPI headers.

A public header may include a UAPI header when that header is self-contained
and does not collide with LLVM-libc's own definitions. When a UAPI header
redefines a type LLVM-libc provides, or defines macros with different
spellings, define the needed constants and types in `llvm-libc-macros/` or
`llvm-libc-types/` instead. For example, UAPI headers that include
`<linux/fcntl.h>` redefine `struct flock` and so cannot be included.

## Dynamic loading

LLVM-libc is linked statically today. Dynamic loading is under development,
mainly so that `dlopen` can work. It is not intended to become the primary way
to use the library, and new code should be designed for the static case first.

## Other operating systems

The points above describe Linux and baremetal targets, where LLVM-libc is the
whole C library. A full build for macOS or Windows has not been designed yet.
The likely direction is for LLVM-libc to define its own API there too, shaped
appropriately for the platform, and not to reproduce POSIX on every system.

## Writing public headers

Public headers are read by every program and by compilers in every language
mode, so they are held to stricter rules than implementation code:

- A header must be valid in every C and C++ standard mode. Do not use language
  features that some of those modes lack, such as attribute syntax (`[[...]]`),
  which C gained only in C23.
- Names that the standard does not define must be reserved identifiers. This
  includes structure members, macro parameters, and helper macros. A member
  named `mutex` or a macro named `NUM_LOCALE_CATEGORIES` collides with user
  code.
- Type headers in `llvm-libc-types/` define one type each and include only the
  other type headers they need. Use the compiler's built-in type macros (for
  example `__INT64_TYPE__`) instead of including `<stdint.h>`.

## Proposing an ABI change

To propose a non-trivial change to the layout of an existing public type or
the value of an existing public macro, or to export a symbol under a name
other than its standard one, contributors should start by reading
{ref}`choosing_type_layout`, then reach out to the maintainers with the
justification and/or write an RFC with the new design. A demo PR may be
helpful for explaining the design, but should not be expected to land without
significant changes.
