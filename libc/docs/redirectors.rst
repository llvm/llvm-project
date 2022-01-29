Redirectors
===========

When implementing a new C standard library (referred to as *libc* henceforth in
this document) starting from scratch, it is unrealistic to expect that we will
have the entire library available from day one. In such a scenario, a practical
approach is to redirect calls to the unimplemented functions to the same
functions from another fully functional libc implementation. Such a scheme can
also serve users who would like to mix and match implementations from LLVM libc
and another libc implementation. On most platforms, this other libc can be the
system libc itself. In this document, we present a strategy one can employ to
build redirectors to redirect from LLVM libc to the system libc. For now, the
scheme presented is limited to ELF platforms.

Highlevel Mechanism
-------------------

The highlevel scheme is as below:

<img src="./redirectors_schematic.svg">

As shown in the diagram, the mechanism involves a redirector dynamic library
which goes in between the llvm-libc static library and the system libc dynamic
library. Essentially, LLVM libc provides implementations for all public
functions. However, some of the implementations do not actually implement the
expected functionality. Instead, they just call the corresponding function in
the redirector library, which in turn calls the same function from the system
libc.

Implementation of redirecting entrypoints
-----------------------------------------

Let us take the ``round`` function from ``math.h`` as an example to see what
it's implementation looks like when it just redirects to the ``round`` function
from the system libc::

    namespace llvm_libc {

    double __redirected_round(double);

    double LLVM_LIBC_ENTRYPOINT(round)(double x) {
        return __redirected_round(x);
    }

    } // namespace llvm_libc

As can be seen, the ``round`` function from LLVM libc does not call the
``round`` function from the system libc directly. It calls a function
``__redirected_round`` from the redirector library. The rest of the
code follows the conventions described in the *implementation standard*
document.

Implementation of the redirector function
-----------------------------------------

The function ``__redirected_round`` calls the ``round`` function from the system
libc. Its implementation is as follows::

    #include <math.h>  // Header file from the system libc

    namespace llvm_libc {

    double __redirected_round(double x) {
        return ::round(x);  // Call to round from the system libc
    }

    } // namespace llvm_libc

