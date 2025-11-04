.. _hand_in_hand:

============
Hand-in-Hand
============

TODO: Docs about hand in hand.

Hand-in-hand is a way for other LLVM projects to use libc's high quality internal APIs.

It's intended to be header only and stable at head, but not necessarily safe for mixing and matching.

Libc++ uses it for from_chars<float>
OpenMP uses it for printf on GPUs
WIP to let clang use it for constexpr math.

External projects shouldn't rely on the interface being stable.
