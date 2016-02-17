============
Using libc++
============

.. contents::
  :local:

Getting Started
===============

If you already have libc++ installed you can use it with clang.

.. code-block:: bash

    $ clang++ -stdlib=libc++ test.cpp
    $ clang++ -std=c++11 -stdlib=libc++ test.cpp

On OS X and FreeBSD libc++ is the default standard library
and the ``-stdlib=libc++`` is not required.

.. _alternate libcxx:

If you want to select an alternate installation of libc++ you
can use the following options.

.. code-block:: bash

  $ clang++ -std=c++11 -stdlib=libc++ -nostdinc++ \
            -I<libcxx-install-prefix>/include/c++/v1 \
            -L<libcxx-install-prefix>/lib \
            -Wl,-rpath,<libcxx-install-prefix>/lib \
            test.cpp

The option ``-Wl,-rpath,<libcxx-install-prefix>/lib`` adds a runtime library
search path. Meaning that the systems dynamic linker will look for libc++ in
``<libcxx-install-prefix>/lib`` whenever the program is run. Alternatively the
environment variable ``LD_LIBRARY_PATH`` (``DYLD_LIBRARY_PATH`` on OS X) can
be used to change the dynamic linkers search paths after a program is compiled.

An example of using ``LD_LIBRARY_PATH``:

.. code-block:: bash

  $ clang++ -stdlib=libc++ -nostdinc++ \
            -I<libcxx-install-prefix>/include/c++/v1
            -L<libcxx-install-prefix>/lib \
            test.cpp -o
  $ ./a.out # Searches for libc++ in the systems library paths.
  $ export LD_LIBRARY_PATH=<libcxx-install-prefix>/lib
  $ ./a.out # Searches for libc++ along LD_LIBRARY_PATH



Using libc++ on Linux
=====================

On Linux libc++ can typically be used with only '-stdlib=libc++'. However
some libc++ installations require the user manually link libc++abi themselves.
If you are running into linker errors when using libc++ try adding '-lc++abi'
to the link line.  For example:

.. code-block:: bash

  $ clang++ -stdlib=libc++ test.cpp -lc++ -lc++abi -lm -lc -lgcc_s -lgcc

Alternately, you could just add libc++abi to your libraries list, which in
most situations will give the same result:

.. code-block:: bash

  $ clang++ -stdlib=libc++ test.cpp -lc++abi


Using libc++ with GCC
---------------------

GCC does not provide a way to switch from libstdc++ to libc++. You must manually
configure the compile and link commands.

In particular you must tell GCC to remove the libstdc++ include directories
using ``-nostdinc++`` and to not link libstdc++.so using ``-nodefaultlibs``.

Note that ``-nodefaultlibs`` removes all of the standard system libraries and
not just libstdc++ so they must be manually linked. For example:

.. code-block:: bash

  $ g++ -nostdinc++ -I<libcxx-install-prefix>/include/c++/v1 \
         test.cpp -nodefaultlibs -lc++ -lc++abi -lm -lc -lgcc_s -lgcc


GDB Pretty printers for libc++
------------------------------

GDB does not support pretty-printing of libc++ symbols by default. Unfortunately
libc++ does not provide pretty-printers itself. However there are 3rd
party implementations available and although they are not officially
supported by libc++ they may be useful to users.

Known 3rd Party Implementations Include:

* `Koutheir's libc++ pretty-printers <https://github.com/koutheir/libcxx-pretty-printers>`_.
