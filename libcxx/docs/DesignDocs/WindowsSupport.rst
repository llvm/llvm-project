===============
Windows support
===============

Currently libc++ needs to link to the MSVC STL ``msvcprt`` library to implement part of the functionality on Windows. This dependency is counterintuitive since one standard library is depending on another standard library. Our goal is to remove this dependency by implementing the required functionality in libc++ itself.

To achieve this, we have decided on the following:

Goals:
------

- ``VCRuntime`` is the underlying ABI layer that libc++ will be using to implement the required functionality for Windows support.

Non-Goals:
----------

- Interoperability with the MSVC STL in the case where both libraries are linked in the same binary is not a goal. For example if you link both libc++ and the MSVC STL in the same binary, a call to a function that needs to maintain internal state like ``std::set_new_handler`` from the libc++ side should not be expected to also change the MSVC STL internal state (since they are separate). We will be maintaining our own internal state for such functions, therefore any such functions should only be expected to have an effect on the libc++ side.
