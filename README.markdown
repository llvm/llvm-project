# The LLVM m88k backend implementation

This repository contains the current state of my **m88k backend for LLVM**.
For general information about LLVM, please read [README.md](README.md).

The Motorola 88000 is a RISC architecture developed by Motorola in the late
1980s. See [Wikipedia](https://en.wikipedia.org/wiki/Motorola_88000) for a
general overview.

The code is based on the example from my book "Learn LLVM 12" (see
[O'Reilly](https://learning.oreilly.com/library/view/learn-llvm-12/9781839213502/),
[Packt](https://www.packtpub.com/product/learn-llvm-12/9781839213502) or
[Amazon](https://www.amazon.com/Learn-LLVM-12-beginners-libraries/dp/1839213507/)).
It differs in the following ways:
- Minimal clang support. Makes it easy to crash the backend.
- All machine instructions are implemented.
- Assembler supports `.requires81100` directive.
- Updates/refactoring of AsmParser, register definition, calling convention, etc.

Future direction:
- Exclusively use new scheduler model. No Itineries!
- Further development concentrates on GlobalISel.

## Building LLVM with the m88k backend

The m88k backend is added as experimental backend to LLVM, named `M88k`.
It's prefined in the CMake files, but if you want to compile other experimental
targets, too, then you have to pass option

```-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD="M88k"```

to CMake.

## Development

The main development happens in branch `main-m88k`. The branch
`release/13.x-m88k` contains an older state based on LLVM 13.x. The branch
`release/12.x-m88k` contains the source from my
[book](https://www.packtpub.com/product/learn-llvm-12/9781839213502) on top of
LLVM 12.x.

This repository mirrors the LLVM `main` and `release/*` branches. I keep my
files on top of `main`, which means that I often rebase. Sorry!

## Support

If you like this work, then there are several ways you can support me:

- You can contributes patches.
- You can provide access to real hardware, e.g. as ssh login.
- If you find some documentation (e.g. the ABI updates from 88open) I would be
  happy to receive a copy.
- You can buy me a beer or a coffee.

## General information about m88k

### Programming manuals

You can find the user manuals for the 88100 and 88110 CPUs, the 88200 MMU and the
88410 secondary cache controller at bitsavers:
http://www.bitsavers.org/components/motorola/88000/. The manual for the 88110
CPU and the 88410 cache controller are also available on GitHub:
https://github.com/awesome-cpus/awesome-cpus-m88k. Some of the files can also be
found on https://archive.org/.

The ELF ABI is only available at
https://archive.org/details/bitsavers_attunixSysa0138776555SystemVRelease488000ABI1990_8011463.

The book [Programming the Motorola 88000](https://www.amazon.com/Programming-Motorola-88000-Michael-Tucker/dp/0830635335/)
by Michael Tucker and Bruce Coorpender (ISBN-13: 978-0830635337,
ISBN-10: 0830635335) is out of print.

In the book [Microprocessor Architectures, 2nd Edition](https://www.oreilly.com/library/view/microprocessor-architectures-2nd/9781483295534/)
by Steve Heath, you find a description of the M88000 architecture in chapter 3.

Regarding the ABI, there were some changes made by the 88open consortium. I have
not yet found that documents. If you have some of them then please contact me!

### Systems

The [OpenBSD/luna88k](https://www.openbsd.org/luna88k.html) port is still active
and provides the latest OS releases.
On a Linux or FreeBSD machine, you can use [GXemul](http://gavare.se/gxemul/) to
run OpenBSD/luna88k.

You can find information about real hardware at [m88k.com](http://m88k.com/).

### Toolchain

gcc 3.3.6 is last version with support for the m88k architecture. See the manual
page at
https://gcc.gnu.org/onlinedocs/gcc-3.3.6/gcc/M88K-Options.html#M88K-Options.

You can download that version from
ftp://ftp.nluug.nl/mirror/languages/gcc/releases/gcc-3.3.6/gcc-3.3.6.tar.gz

binutils 2.16 is last version with support for the m88k architecture. See the
manual page at
https://sourceware.org/binutils/docs-2.16/
