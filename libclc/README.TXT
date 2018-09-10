libclc
------

libclc is an open source, BSD licensed implementation of the library
requirements of the OpenCL C programming language, as specified by the
OpenCL 1.1 Specification. The following sections of the specification
impose library requirements:

  * 6.1: Supported Data Types
  * 6.2.3: Explicit Conversions
  * 6.2.4.2: Reinterpreting Types Using as_type() and as_typen()
  * 6.9: Preprocessor Directives and Macros
  * 6.11: Built-in Functions
  * 9.3: Double Precision Floating-Point
  * 9.4: 64-bit Atomics
  * 9.5: Writing to 3D image memory objects
  * 9.6: Half Precision Floating-Point

libclc is intended to be used with the Clang compiler's OpenCL frontend.

libclc is designed to be portable and extensible. To this end, it provides
generic implementations of most library requirements, allowing the target
to override the generic implementation at the granularity of individual
functions.

libclc currently only supports the PTX target, but support for more
targets is welcome.

Compiling and installing with Make
----------------------------------

$ ./configure.py --with-llvm-config=/path/to/llvm-config && make
$ make install

Note you can use the DESTDIR Makefile variable to do staged installs.

$ make install DESTDIR=/path/for/staged/install

Compiling and installing with Ninja
-----------------------------------

$ ./configure.py -g ninja --with-llvm-config=/path/to/llvm-config && ninja
$ ninja install

Note you can use the DESTDIR environment variable to do staged installs.

$ DESTDIR=/path/for/staged/install ninja install

Website
-------

http://www.pcc.me.uk/~peter/libclc/
