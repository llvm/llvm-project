==========================
Layering Over Another libc
==========================

When meaningful and practically possible on a platform, llvm-libc will be
developed in a fashion that it will be possible to layer it over the system
libc. This does not mean that one can mix llvm-libc with the system-libc. Also,
it does not mean that layering is the only way to use llvm-libc. What it
means is that, llvm-libc can optionally be packaged in a way that it can
delegate parts of the functionality to the system-libc. The delegation happens
internal to llvm-libc and is invisible to the users. From the user's point of
view, they only call into llvm-libc.

There are a few problems one needs to be mindful of when implementing such a
delegation scheme in llvm-libc. Examples of such problems are:

1. One cannot mix data structures from llvm-libc with those from the
system-libc. A translation from one set of data structures to the other should
happen internal to llvm-libc.
2. The delegation mechanism has to be implemented over a related set of
functions. For example, one cannot delegate just the `fopen` function to the
system-libc. One will have to delegate all `FILE` related functions to the
system-libc.
