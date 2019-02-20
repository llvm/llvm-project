//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <dlfcn.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <stdlib.h>

int
main (int argc, char const *argv[])
{
#if defined (__APPLE__)
    const char *a_name = "@executable_path/libloadunload_a.dylib";
    const char *c_name = "@executable_path/libloadunload_c.dylib";
#else
    const char *a_name = "libloadunload_a.so";
    const char *c_name = "libloadunload_c.so";
#endif
    void *a_dylib_handle = NULL;
    void *c_dylib_handle = NULL;
    int (*a_function) (void);

    a_dylib_handle = dlopen (a_name, RTLD_NOW); // Set break point at this line for test_lldb_process_load_and_unload_commands().
    if (a_dylib_handle == NULL)
    {
        fprintf (stderr, "%s\n", dlerror());
        exit (1);
    }

    a_function = (int (*) ()) dlsym (a_dylib_handle, "a_function");
    if (a_function == NULL)
    {
        fprintf (stderr, "%s\n", dlerror());
        exit (2);
    }
    printf ("First time around, got: %d\n", a_function ());
    dlclose (a_dylib_handle);

    c_dylib_handle = dlopen (c_name, RTLD_NOW);
    if (c_dylib_handle == NULL)
    {
        fprintf (stderr, "%s\n", dlerror());
        exit (3);
    }
    a_function = (int (*) ()) dlsym (c_dylib_handle, "c_function");
    if (a_function == NULL)
    {
        fprintf (stderr, "%s\n", dlerror());
        exit (4);
    }

    a_dylib_handle = dlopen (a_name, RTLD_NOW);
    if (a_dylib_handle == NULL)
    {
        fprintf (stderr, "%s\n", dlerror());
        exit (5);
    }

    a_function = (int (*) ()) dlsym (a_dylib_handle, "a_function");
    if (a_function == NULL)
    {
        fprintf (stderr, "%s\n", dlerror());
        exit (6);
    }
    printf ("Second time around, got: %d\n", a_function ());
    dlclose (a_dylib_handle);

    int d_function(void);
    printf ("d_function returns: %d\n", d_function());

    return 0;
}
