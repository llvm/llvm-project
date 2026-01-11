.. _hand_in_hand:

============
Hand-in-Hand
============

Hand-in-Hand is the name of the mechanism that allows other LLVM projects to use
LLVM-libc's internal C++ APIs instead of calling the public libc interface.
This is useful for cases where the C interface doesn't match the desired
interface.

The original use case for the Hand-in-Hand interface was to let libc++ use
LLVM-libc's string to float conversion internals. The libc interface (strtof)
takes a null terminated string with no maximum length while the libc++ interface
(from_chars<float>) takes a string with a start and an end. If libc++ had used
the public interface it would have had to allocate a new null terminated string
before calling strtof, but with Hand-in-Hand libc++ handles its own parsing
and then passes the parsed information to LLVM-libc's conversion code. This is
better for performance and cuts down on code duplication in the LLVM repository.

Hand-in-Hand works by LLVM-libc exposing a set of headers in the /libc/shared/
directory. These headers make the interface more explicit and easier to
maintain. The client library includes the shared headers by depending on the
llvm-libc-common-utilities target which sets up the necessary includes and
defines. The client library then includes "shared/<header>" to get the necessary
components. All of the functions shared via Hand-in-Hand are header only.

The Hand-in-Hand interface is intended to be an internal implementation detail,
and it has no guarantees of stability. When the internal LLVM-libc interface is
changed the other users inside of the LLVM repository are updated in the same
commit. This allows LLVM-libc to update their interface without breaking their
users.

Current Hand-in-Hand users:
Libc++ uses it for from_chars<float/double>
OpenMP uses it for printf on GPUs
[WIP] clang uses it for APFloat functions.

For more information check out the 2024 talk about the original Project:
  * `slides <https://llvm.org/devmtg/2024-10/slides/techtalk/Jones-DiBella-hand-in-hand.pdf>`__
  * `video <https://www.youtube.com/watch?v=VAEO86YtTHA>`__
