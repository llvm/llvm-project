# The LLVM libc unit test framework

This directory contains a lightweight implementation of a
[gtest](https://github.com/google/googletest) like unit test framework for LLVM
libc.

## Why not gtest?

While gtest is great, featureful and time tested, it uses the C and C++
standard libraries. Hence, using it to test LLVM libc (which is also an
implementation of the C standard libraries) causes various kinds of
mixup/conflict problems.

## How is it different from gtest?

LLVM libc's unit test framework is much less featureful as compared to gtest.
But, what is available strives to be exactly like gtest.

## Will it be made as featureful as gtest in future?

It is not clear if LLVM libc needs/will need every feature of gtest. We only
intend to extend it on an _as needed_ basis. Hence, it might never be as
featureful as gtest.
