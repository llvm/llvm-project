/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

char **__argv_save;
int __argc_save;

/* get saved argv */

char **
__io_get_argv()
{
  return (__argv_save);
}

/* set saved argv */

void
__io_set_argv(char **v)
{
  __argv_save = v;
}

/* get saved argc */

int
__io_get_argc()
{
  return (__argc_save);
}

/* set saved argc */

void
__io_set_argc(int v)
{
  __argc_save = v;
}

