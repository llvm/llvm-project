Examples
========
This directory contains a few example programs which illustrate how one can set
up their own projects to use LLVM's libc, either as an overlay or as the only
libc in their projects. See the
[the usage mode document](https://libc.llvm.org/usage_modes.html) for more
information about the different modes in which one can build and use the libc.

Building the Examples
=====================
Each example has its own directory which contain the source code and the CMake
build set up. To build an example, create a directory named `build` in the
example's directory:

```bash
cd <example directory>
mkdir build
cd build
```

Each example can be built to use the libc in either
[the overlay mode](https://libc.llvm.org/overlay_mode.html) or the
[full build mode](https://libc.llvm.org/fullbuild_mode.html). The CMake
configure step differs slightly depending on the mode you want to use the libc
in.

Building against an overlay libc
--------------------------------

Before you can link an example against the overlay libc, you will have to
install it. See [the documentation of the overlay mode](https://libc.llvm.org/overlay_mode.html)
to learn how to install the libc's overlay static archive named `libllvmlibc.a`.
Once installed, to build an example against it, you have specify the directory
in which the static archive is installed with the option
`LIBC_OVERLAY_ARCHIVE_DIR`:

```bash
cmake ../ -G <GEN>  \
  -DLIBC_OVERLAY_ARCHIVE_DIR=<dir in which libc is installed>
```

Next, if `Ninja` is used for `<GEN>`, you can build the example as follows:

```bash
ninja <example name>
```

Building against a full libc
----------------------------

Before you can link an example against the full libc, you will have to first
install it. See [the documentation of the full build mode](https://libc.llvm.org/fullbuild_mode.html)
to learn how to install a full libc along with the other LLVM toolchain pieces
like `clang`, `lld` and `compiler-rt`. The CMake build for the examples will
assume that you have all of these components installed in a special sysroot
(see decription of the `--sysroot` option
[here](https://gcc.gnu.org/onlinedocs/gcc/Directory-Options.html).) Once you
have installed them, you have to inform CMake that we are linking against the
full libc as follows:

```bash
cmake ../ -G <GEN> -DLLVM_LIBC_FULL_BUILD=ON    \
  -DCMAKE_SYSROOT=<SYSROOT>               \
  -DCMAKE_C_COMPILER=<SYSROOT>/bin/clang  \
  -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY
```

`<SYSROOT>` is the path to the sysroot directory you have set up while
installing the full libc. The option
`-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY` tells CMake to not attempt
linking full executables against shared libraries. We have to use this as LLVM's
libc does not yet have support for shared libraries and dynamic linking. After
the above `cmake` command, assuming `Ninja` was used for `<GEN>`, you can build
the example as follows:


```bash
ninja <example name>
```
