The mechanics of the ``public_api`` command
===========================================

The build system, in combination with the header generation mechanism,
facilitates the fine grained ability to pick and choose the public API one wants
to expose on their platform. The public header files are always generated from
the corresponding ``.h.def`` files. A header generation command ``%%public_api``
is listed in these files. In the generated header file, the header generator
replaces this command with the public API relevant for the target platform.

Under the hood
--------------

When the header generator sees the ``%%public_api`` command, it looks up the
API config file for the platform in the path ``config/<platform>/api.td``.
The API config file lists two kinds of items:

1. The list of standards from which the public entities available on the platform
   are derived from.
2. For each header file exposed on the platform, the list of public members
   provided in that header file.

Note that, the header generator only learns the names of the public entities
from the header config file (the 2nd item from above.) The exact manner in which
the entities are to be declared is got from the standards (the 1st item from
above.)

See the ground truth document for more information on how the standards are
formally listed in LLVM libc using LLVM table-gen files.
