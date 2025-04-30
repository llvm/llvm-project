/* The internal wrapper of clone and clone3.
   Copyright (C) 2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include <stddef.h>
#include <errno.h>
#include <sched.h>
#include <clone_internal.h>
#include <libc-pointer-arith.h>	/* For cast_to_pointer.  */
#include <stackinfo.h>		/* For _STACK_GROWS_{UP,DOWN}.  */

#define CLONE_ARGS_SIZE_VER0 64 /* sizeof first published struct */
#define CLONE_ARGS_SIZE_VER1 80 /* sizeof second published struct */
#define CLONE_ARGS_SIZE_VER2 88 /* sizeof third published struct */

#define sizeof_field(TYPE, MEMBER) sizeof ((((TYPE *)0)->MEMBER))
#define offsetofend(TYPE, MEMBER) \
  (offsetof (TYPE, MEMBER) + sizeof_field (TYPE, MEMBER))

_Static_assert (__alignof (struct clone_args) == 8,
		"__alignof (struct clone_args) != 8");
_Static_assert (offsetofend (struct clone_args, tls) == CLONE_ARGS_SIZE_VER0,
		"offsetofend (struct clone_args, tls) != CLONE_ARGS_SIZE_VER0");
_Static_assert (offsetofend (struct clone_args, set_tid_size) == CLONE_ARGS_SIZE_VER1,
		"offsetofend (struct clone_args, set_tid_size) != CLONE_ARGS_SIZE_VER1");
_Static_assert (offsetofend (struct clone_args, cgroup) == CLONE_ARGS_SIZE_VER2,
		"offsetofend (struct clone_args, cgroup) != CLONE_ARGS_SIZE_VER2");
_Static_assert (sizeof (struct clone_args) == CLONE_ARGS_SIZE_VER2,
		"sizeof (struct clone_args) != CLONE_ARGS_SIZE_VER2");

int
__clone_internal (struct clone_args *cl_args,
		  int (*func) (void *arg), void *arg)
{
  int ret;

  /* Map clone3 arguments to clone arguments.  NB: No need to check
     invalid clone3 specific bits in flags nor exit_signal since this
     is an internal function.  */
  int flags = cl_args->flags | cl_args->exit_signal;
  void *stack = cast_to_pointer (cl_args->stack);

#ifdef __ia64__
  ret = __clone2 (func, stack, cl_args->stack_size,
		  flags, arg,
		  cast_to_pointer (cl_args->parent_tid),
		  cast_to_pointer (cl_args->tls),
		  cast_to_pointer (cl_args->child_tid));
#else
# if !_STACK_GROWS_DOWN && !_STACK_GROWS_UP
#  error "Define either _STACK_GROWS_DOWN or _STACK_GROWS_UP"
# endif

# if _STACK_GROWS_DOWN
  stack += cl_args->stack_size;
# endif
  ret = __clone (func, stack, flags, arg,
		 cast_to_pointer (cl_args->parent_tid),
		 cast_to_pointer (cl_args->tls),
		 cast_to_pointer (cl_args->child_tid));
#endif
  return ret;
}

libc_hidden_def (__clone_internal)
