=============
Configuration
=============

.. contents::

.. role:: raw-html(raw)
   :format: html

Clangd has a bunch of command-line options that can change its behaviour in
certain situations. This page aims to define those configuration knobs.

Those command line arguments needs to be specified in an editor-specific way.
You can find some editor specific instructions in `here <https://clang.llvm.org/extra/clangd/Installation.html#id3>`__.

--query-driver
==============

Clangd makes use of clang behind the scenes, so it might fail to detect your
standard library or built-in headers if your project is making use of a custom
toolchain. That is quite common in hardware-related projects, especially for the
ones making use of gcc (e.g. ARM's `arm-none-eabi-gcc`).

You can specify your driver as a list of globs or full paths, then clangd will
execute drivers and fetch necessary include paths to compile your code.

For example if you have your compilers at:
 - `/path/to/my-custom/toolchain1/arm-none-eabi-gcc`,
 - `/path/to/my-custom/toolchain2/arm-none-eabi-g++`,
 - `/path/to/my-custom2/toolchain/arm-none-eabi-g++`,
   
you can provide clangd with
`--query-driver=/path/to/my-custom/**/arm-none-eabi*` to enable execution of
any binary that has a name starting with `arm-none-eabi` and under
`/path/to/my-custom/`. This won't allow execution of the last compiler.

Full list of flags
==================

You can find out about the rest of the flags using `clangd --help`.
