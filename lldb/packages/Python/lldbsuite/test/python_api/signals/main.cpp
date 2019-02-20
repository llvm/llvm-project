//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <sys/types.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <signal.h>
#endif

// This simple program is to test the lldb Python API related to process.

int main (int argc, char const *argv[])
{
#if defined(_WIN32)
		::ExitProcess(1);
#else
    kill(getpid(), SIGINT); // Set break point at this line and setup signal ignores.
#endif
    return 0;
}
