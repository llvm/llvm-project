/* Copyright (C) 2014-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sys/prctl.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <errno.h>

#if defined PR_GET_FP_MODE && defined PR_SET_FP_MODE
# define HAVE_PRCTL_FP_MODE 1
# define FR1_MODE (PR_FP_MODE_FR)
# define FRE_MODE (PR_FP_MODE_FR | PR_FP_MODE_FRE)
#else
# define HAVE_PRCTL_FP_MODE 0
# define FR1_MODE 0x1
# define FRE_MODE 0x2
#endif

#define STR_VAL(VAL) #VAL
#define N_STR(VAL) STR_VAL(VAL)

#define START_STATE(NAME) 					\
case s_ ## NAME: 						\
  {								\
    switch (obj) 						\
      {

#define END_STATE						\
      default:							\
        return false;						\
      }								\
  break;							\
  }

#define NEXT(OBJ, NEXT_STATE)					\
case o_ ## OBJ: 						\
  current_fp_state = s_ ## NEXT_STATE;				\
  break;

#define NEXT_REQ_FR1(OBJ, NEXT_STATE)				\
case o_ ## OBJ:							\
  {								\
    if (has_fr1)						\
      current_fp_state = s_ ## NEXT_STATE;			\
    else							\
      return false;						\
  }								\
  break;

#define NEXT_REQ_FR0(OBJ, NEXT_STATE) 				\
case o_ ## OBJ:							\
  {								\
    if (!is_r6							\
        || (is_r6 && has_fr1 && has_fre))			\
      current_fp_state = s_ ## NEXT_STATE;			\
    else 							\
      return false;						\
  }								\
  break;

#define NEXT_REQ_FRE(OBJ, NEXT_STATE)				\
case o_ ## OBJ: 						\
  {								\
    if (has_fr1 && has_fre)					\
      current_fp_state = s_ ## NEXT_STATE;			\
    else							\
      return false;						\
  }								\
  break;

#define NEXT_NO_MODE_CHANGE(OBJ, NEXT_STATE)			\
case o_ ## OBJ: 						\
  {								\
    if (current_mode_valid_p (s_ ## NEXT_STATE))			\
      {								\
	current_fp_state = s_ ## NEXT_STATE;			\
	cant_change_mode = true;				\
      }								\
    else							\
      return false;						\
  }								\
  break;

static const char * const shared_lib_names[] =
  {
    "tst-abi-fpanymod.so", "tst-abi-fpsoftmod.so", "tst-abi-fpsinglemod.so",
    "tst-abi-fp32mod.so", "tst-abi-fp64mod.so", "tst-abi-fp64amod.so",
    "tst-abi-fpxxmod.so", "tst-abi-fpxxomod.so"
  };

struct fp_mode_req
{
  int mode1;
  int mode2;
  int mode3;
};

enum fp_obj
{
  o_any,
  o_soft,
  o_single,
  o_fp32,
  o_fp64,
  o_fp64a,
  o_fpxx,
  o_fpxxo,
  o_max
};

enum fp_state
{
  s_any,
  s_soft,
  s_single,
  s_fp32,
  s_fpxx,
  s_fpxxo,
  s_fp64a,
  s_fp64,
  s_fpxxo_fpxx,
  s_fp32_fpxx,
  s_fp32_fpxxo,
  s_fp32_fpxxo_fpxx,
  s_fp32_fp64a_fpxx,
  s_fp32_fp64a_fpxxo,
  s_fp32_fp64a_fpxxo_fpxx,
  s_fp64a_fp32,
  s_fp64a_fpxx,
  s_fp64a_fpxxo,
  s_fp64a_fp64,
  s_fp64a_fp64_fpxx,
  s_fp64a_fp64_fpxxo,
  s_fp64a_fpxx_fpxxo,
  s_fp64a_fp64_fpxxo_fpxx,
  s_fp64_fpxx,
  s_fp64_fpxxo,
  s_fp64_fpxx_fpxxo
};


static int current_fp_mode;
static bool cant_change_mode = false;
static bool has_fr1 = false;
static bool has_fre = false;
static bool is_r6 = false;
static unsigned int fp_obj_count[o_max];
void * shared_lib_ptrs[o_max];
static enum fp_state current_fp_state = s_any;
static enum fp_obj test_objects[FPABI_COUNT] = { FPABI_LIST };

/* This function will return the valid FP modes for the specified state.  */

static struct fp_mode_req
compute_fp_modes (enum fp_state state)
{
  struct fp_mode_req requirements;

  requirements.mode1 = -1;
  requirements.mode2 = -1;
  requirements.mode3 = -1;

  switch (state)
    {
    case s_single:
      {
        if (is_r6)
	  requirements.mode1 = FR1_MODE;
	else
	  {
	    requirements.mode1 = 0;
	    requirements.mode2 = FR1_MODE;
	  }
	break;
      }
    case s_fp32:
    case s_fp32_fpxx:
    case s_fp32_fpxxo:
    case s_fp32_fpxxo_fpxx:
      {
	if (is_r6)
	  requirements.mode1 = FRE_MODE;
	else
	  {
	    requirements.mode1 = 0;
	    requirements.mode2 = FRE_MODE;
	  }
	break;
      }
    case s_fpxx:
    case s_fpxxo:
    case s_fpxxo_fpxx:
    case s_any:
    case s_soft:
      {
	if (is_r6)
	  {
	    requirements.mode1 = FR1_MODE;
	    requirements.mode2 = FRE_MODE;
	  }
	else
	  {
	    requirements.mode1 = 0;
	    requirements.mode2 = FR1_MODE;
	    requirements.mode3 = FRE_MODE;
	  }
	break;
      }
    case s_fp64a:
    case s_fp64a_fpxx:
    case s_fp64a_fpxxo:
    case s_fp64a_fpxx_fpxxo:
      {
	requirements.mode1 = FR1_MODE;
	requirements.mode2 = FRE_MODE;
	break;
      }
    case s_fp64:
    case s_fp64_fpxx:
    case s_fp64_fpxxo:
    case s_fp64_fpxx_fpxxo:
    case s_fp64a_fp64:
    case s_fp64a_fp64_fpxx:
    case s_fp64a_fp64_fpxxo:
    case s_fp64a_fp64_fpxxo_fpxx:
      {
	requirements.mode1 = FR1_MODE;
	break;
      }
    case s_fp64a_fp32:
    case s_fp32_fp64a_fpxx:
    case s_fp32_fp64a_fpxxo:
    case s_fp32_fp64a_fpxxo_fpxx:
      {
        requirements.mode1 = FRE_MODE;
        break;
      }
    }
  return requirements;
}

/* Check the current mode is suitable for the specified state.  */

static bool
current_mode_valid_p (enum fp_state s)
{
  struct fp_mode_req req = compute_fp_modes (s);
  return (req.mode1 == current_fp_mode
	  || req.mode2 == current_fp_mode
	  || req.mode3 == current_fp_mode);
}

/* Run the state machine by adding a new object.  */

static bool
set_next_fp_state (enum fp_obj obj)
{
  cant_change_mode = false;
  switch (current_fp_state)
    {

    START_STATE(soft)
    NEXT(soft,soft)
    NEXT(any,soft)
    END_STATE

    START_STATE(single)
    NEXT(single,single)
    NEXT(any,single)
    END_STATE

    START_STATE(any)
    NEXT_REQ_FR0(fp32, fp32)
    NEXT(fpxx, fpxx)
    NEXT(fpxxo, fpxxo)
    NEXT_REQ_FR1(fp64a, fp64a)
    NEXT_REQ_FR1(fp64, fp64)
    NEXT(any,any)
    NEXT(soft,soft)
    NEXT(single,single)
    END_STATE

    START_STATE(fp32)
    NEXT_REQ_FR0(fp32,fp32)
    NEXT(fpxx, fp32_fpxx)
    NEXT(fpxxo, fp32_fpxxo)
    NEXT_REQ_FRE(fp64a, fp64a_fp32)
    NEXT(any,fp32)
    END_STATE

    START_STATE(fpxx)
    NEXT_REQ_FR0(fp32, fp32_fpxx)
    NEXT_REQ_FR1(fp64, fp64_fpxx)
    NEXT_REQ_FR1(fp64a, fp64a_fpxx)
    NEXT(fpxxo, fpxxo_fpxx)
    NEXT(fpxx,fpxx)
    NEXT(any,fpxx)
    END_STATE

    START_STATE(fpxxo)
    NEXT_NO_MODE_CHANGE(fp32, fp32_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64, fp64_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64a, fp64a_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxx, fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fpxxo,fpxxo)
    NEXT_NO_MODE_CHANGE(any,fpxxo)
    END_STATE

    START_STATE(fp64a)
    NEXT_REQ_FRE(fp32, fp64a_fp32)
    NEXT_REQ_FR1(fp64, fp64a_fp64)
    NEXT(fpxxo, fp64a_fpxxo)
    NEXT(fpxx, fp64a_fpxx)
    NEXT_REQ_FR1(fp64a, fp64a)
    NEXT(any, fp64a)
    END_STATE

    START_STATE(fp64)
    NEXT_REQ_FR1(fp64a, fp64a_fp64)
    NEXT(fpxxo, fp64_fpxxo)
    NEXT(fpxx, fp64_fpxx)
    NEXT_REQ_FR1(fp64, fp64)
    NEXT(any, fp64)
    END_STATE

    START_STATE(fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fp32, fp32_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fp64, fp64_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64a, fp64a_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxx, fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fpxxo, fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(any, fpxxo_fpxx)
    END_STATE

    START_STATE(fp32_fpxx)
    NEXT_REQ_FR0(fp32, fp32_fpxx)
    NEXT(fpxx, fp32_fpxx)
    NEXT(fpxxo, fp32_fpxxo_fpxx)
    NEXT_REQ_FRE(fp64a, fp32_fp64a_fpxx)
    NEXT(any, fp32_fpxx)
    END_STATE

    START_STATE(fp32_fpxxo)
    NEXT_NO_MODE_CHANGE(fp32, fp32_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxxo, fp32_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxx, fp32_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fp64a, fp32_fp64a_fpxxo)
    NEXT_NO_MODE_CHANGE(any, fp32_fpxxo)
    END_STATE

    START_STATE(fp32_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fp32, fp32_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fpxxo, fp32_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fpxx, fp32_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fp64a, fp32_fp64a_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(any, fp32_fpxxo_fpxx)
    END_STATE

    START_STATE(fp64a_fp32)
    NEXT_REQ_FRE(fp32, fp64a_fp32)
    NEXT_REQ_FRE(fp64a, fp64a_fp32)
    NEXT(fpxxo, fp32_fp64a_fpxxo)
    NEXT(fpxx, fp32_fp64a_fpxx)
    NEXT(any, fp64a_fp32)
    END_STATE

    START_STATE(fp64a_fpxx)
    NEXT_REQ_FRE(fp32, fp32_fp64a_fpxx)
    NEXT_REQ_FR1(fp64a, fp64a_fpxx)
    NEXT(fpxx, fp64a_fpxx)
    NEXT(fpxxo, fp64a_fpxx_fpxxo)
    NEXT_REQ_FR1(fp64, fp64a_fp64_fpxx)
    NEXT(any, fp64a_fpxx)
    END_STATE

    START_STATE(fp64a_fpxxo)
    NEXT_NO_MODE_CHANGE(fp32, fp32_fp64a_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64a, fp64a_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxx, fp64a_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxxo, fp64a_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64, fp64a_fp64_fpxxo)
    NEXT_NO_MODE_CHANGE(any, fp64a_fpxxo)
    END_STATE

    START_STATE(fp64a_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(fp32, fp32_fp64a_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fp64a, fp64a_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxx, fp64a_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxxo, fp64a_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64, fp64a_fp64_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(any, fp64a_fpxx_fpxxo)
    END_STATE

    START_STATE(fp64_fpxx)
    NEXT_REQ_FR1(fp64a, fp64a_fp64_fpxx)
    NEXT(fpxxo, fp64_fpxx_fpxxo)
    NEXT(fpxx, fp64_fpxx)
    NEXT_REQ_FR1(fp64, fp64_fpxx)
    NEXT(any, fp64_fpxx)
    END_STATE

    START_STATE(fp64_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64a, fp64a_fp64_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxxo, fp64_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxx, fp64_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64, fp64_fpxxo)
    NEXT_NO_MODE_CHANGE(any, fp64_fpxxo)
    END_STATE

    START_STATE(fp64_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64a, fp64a_fp64_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fpxxo, fp64_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxx, fp64_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64, fp64_fpxx_fpxxo)
    NEXT_NO_MODE_CHANGE(any, fp64_fpxx_fpxxo)
    END_STATE

    START_STATE(fp64a_fp64)
    NEXT_REQ_FR1(fp64a, fp64a_fp64)
    NEXT(fpxxo, fp64a_fp64_fpxxo)
    NEXT(fpxx, fp64a_fp64_fpxx)
    NEXT_REQ_FR1(fp64, fp64a_fp64)
    NEXT(any, fp64a_fp64)
    END_STATE

    START_STATE(fp64a_fp64_fpxx)
    NEXT_REQ_FR1(fp64a, fp64a_fp64_fpxx)
    NEXT(fpxxo, fp64a_fp64_fpxxo_fpxx)
    NEXT(fpxx, fp64a_fp64_fpxx)
    NEXT_REQ_FR1(fp64, fp64a_fp64_fpxx)
    NEXT(any, fp64a_fp64_fpxx)
    END_STATE

    START_STATE(fp64a_fp64_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64a, fp64a_fp64_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxx, fp64a_fp64_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fpxxo, fp64a_fp64_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64, fp64a_fp64_fpxxo)
    NEXT_NO_MODE_CHANGE(any, fp64a_fp64_fpxxo)
    END_STATE

    START_STATE(fp64a_fp64_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fp64a, fp64a_fp64_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fpxx, fp64a_fp64_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fpxxo, fp64a_fp64_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fp64, fp64a_fp64_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(any, fp64a_fp64_fpxxo_fpxx)
    END_STATE

    START_STATE(fp32_fp64a_fpxx)
    NEXT_REQ_FRE(fp32, fp32_fp64a_fpxx)
    NEXT_REQ_FRE(fp64a, fp32_fp64a_fpxx)
    NEXT(fpxxo, fp32_fp64a_fpxxo_fpxx)
    NEXT(fpxx, fp32_fp64a_fpxx)
    NEXT(any, fp32_fp64a_fpxx)
    END_STATE

    START_STATE(fp32_fp64a_fpxxo)
    NEXT_NO_MODE_CHANGE(fp32, fp32_fp64a_fpxxo)
    NEXT_NO_MODE_CHANGE(fp64a, fp32_fp64a_fpxxo)
    NEXT_NO_MODE_CHANGE(fpxx, fp32_fp64a_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fpxxo, fp32_fp64a_fpxxo)
    NEXT_NO_MODE_CHANGE(any, fp32_fp64a_fpxxo)
    END_STATE

    START_STATE(fp32_fp64a_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fp32, fp32_fp64a_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fp64a, fp32_fp64a_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fpxx, fp32_fp64a_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(fpxxo, fp32_fp64a_fpxxo_fpxx)
    NEXT_NO_MODE_CHANGE(any, fp32_fp64a_fpxxo_fpxx)
    END_STATE
    }

  if (obj != o_max)
    fp_obj_count[obj]++;

  return true;
}

/* Run the state machine by removing an object.  */

static bool
remove_object (enum fp_obj obj)
{
  if (obj == o_max)
    return false;

  fp_obj_count[obj]--;

  /* We can't change fp state until all the objects
     of a particular type have been unloaded.  */
  if (fp_obj_count[obj] != 0)
    return false;

  switch (current_fp_state)
    {
    START_STATE(soft)
    NEXT(soft,any)
    END_STATE

    START_STATE(single)
    NEXT(single,any)
    END_STATE

    START_STATE(any)
    NEXT(any,any)
    END_STATE

    START_STATE(fp32)
    NEXT (fp32,any)
    END_STATE

    START_STATE(fpxx)
    NEXT (fpxx,any)
    END_STATE

    START_STATE(fpxxo)
    NEXT (fpxxo,any)
    END_STATE

    START_STATE(fp64a)
    NEXT(fp64a, any)
    END_STATE

    START_STATE(fp64)
    NEXT(fp64, any)
    END_STATE

    START_STATE(fpxxo_fpxx)
    NEXT(fpxx, fpxxo)
    NEXT(fpxxo, fpxx)
    END_STATE

    START_STATE(fp32_fpxx)
    NEXT(fp32, fpxx)
    NEXT(fpxx, fp32)
    END_STATE

    START_STATE(fp32_fpxxo)
    NEXT(fp32, fpxxo)
    NEXT(fpxxo, fp32)
    END_STATE

    START_STATE(fp32_fpxxo_fpxx)
    NEXT(fp32, fpxxo_fpxx)
    NEXT(fpxxo, fp32_fpxx)
    NEXT(fpxx, fp32_fpxxo)
    END_STATE

    START_STATE(fp64a_fp32)
    NEXT(fp32, fp64a)
    NEXT(fp64a, fp32)
    END_STATE

    START_STATE(fp64a_fpxx)
    NEXT(fp64a, fpxx)
    NEXT(fpxx, fp64a)
    END_STATE

    START_STATE(fp64a_fpxxo)
    NEXT(fp64a, fpxxo)
    NEXT(fpxxo, fp64a)
    END_STATE

    START_STATE(fp64a_fpxx_fpxxo)
    NEXT(fp64a, fpxxo_fpxx)
    NEXT(fpxx, fp64a_fpxxo)
    NEXT(fpxxo, fp64a_fpxx)
    END_STATE

    START_STATE(fp64_fpxx)
    NEXT(fpxx, fp64)
    NEXT(fp64, fpxx)
    END_STATE

    START_STATE(fp64_fpxxo)
    NEXT(fpxxo, fp64)
    NEXT(fp64, fpxxo)
    END_STATE

    START_STATE(fp64_fpxx_fpxxo)
    NEXT(fp64, fpxxo_fpxx)
    NEXT(fpxxo, fp64_fpxx)
    NEXT(fpxx, fp64_fpxxo)
    END_STATE

    START_STATE(fp64a_fp64)
    NEXT(fp64a, fp64)
    NEXT(fp64, fp64a)
    END_STATE

    START_STATE(fp64a_fp64_fpxx)
    NEXT(fp64a, fp64_fpxx)
    NEXT(fpxx, fp64a_fp64)
    NEXT(fp64, fp64a_fpxx)
    END_STATE

    START_STATE(fp64a_fp64_fpxxo)
    NEXT(fp64a, fp64_fpxxo)
    NEXT(fpxxo, fp64a_fp64)
    NEXT(fp64, fp64a_fpxxo)
    END_STATE

    START_STATE(fp64a_fp64_fpxxo_fpxx)
    NEXT(fp64a, fp64_fpxx_fpxxo)
    NEXT(fpxx, fp64a_fp64_fpxxo)
    NEXT(fpxxo, fp64a_fp64_fpxx)
    NEXT(fp64, fp64a_fpxx_fpxxo)
    END_STATE

    START_STATE(fp32_fp64a_fpxx)
    NEXT(fp32, fp64a_fpxx)
    NEXT(fp64a, fp32_fpxx)
    NEXT(fpxx, fp64a_fp32)
    END_STATE

    START_STATE(fp32_fp64a_fpxxo)
    NEXT(fp32, fp64a_fpxxo)
    NEXT(fp64a, fp32_fpxxo)
    NEXT(fpxxo, fp64a_fp32)
    END_STATE

    START_STATE(fp32_fp64a_fpxxo_fpxx)
    NEXT(fp32, fp64a_fpxx_fpxxo)
    NEXT(fp64a, fp32_fpxxo_fpxx)
    NEXT(fpxx, fp32_fp64a_fpxxo)
    NEXT(fpxxo, fp32_fp64a_fpxx)
    END_STATE
    }

  return true;
}

static int
mode_transition_valid_p (void)
{
  int prev_fp_mode;

  /* Get the current fp mode.  */
  prev_fp_mode = current_fp_mode;
#if HAVE_PRCTL_FP_MODE
  current_fp_mode = prctl (PR_GET_FP_MODE);

  /* If the prctl call fails assume the core only has FR0 mode support.  */
  if (current_fp_mode == -1)
    current_fp_mode = 0;
#endif

  if (!current_mode_valid_p (current_fp_state))
    return 0;

  /* Check if mode changes are not allowed but a mode change happened.  */
  if (cant_change_mode
      && current_fp_mode != prev_fp_mode)
    return 0;

  return 1;
}

/* Load OBJ and check that it was/was not loaded correctly.  */
bool
load_object (enum fp_obj obj)
{
  bool should_load = set_next_fp_state (obj);

  shared_lib_ptrs[obj] = dlopen (shared_lib_names[obj], RTLD_LAZY);

  /* If we expected an error and the load was successful then fail.  */
  if (!should_load && (shared_lib_ptrs[obj] != 0))
    return false;

  if (should_load && (shared_lib_ptrs[obj] == 0))
    return false;

  if (!mode_transition_valid_p ())
    return false;

  return true;
}

/* Remove an object and check the state remains valid.  */
bool
unload_object (enum fp_obj obj)
{
  if (!shared_lib_ptrs[obj])
    return true;

  remove_object (obj);

  if (dlclose (shared_lib_ptrs[obj]) != 0)
    return false;

  shared_lib_ptrs[obj] = 0;

  if (!mode_transition_valid_p ())
    return false;

  return true;
}

/* Load every permuation of OBJECTS.  */
static bool
test_permutations (enum fp_obj objects[], int count)
{
  int i;

  for (i = 0 ; i < count ; i++)
    {
      if (!load_object (objects[i]))
	return false;

      if (count > 1)
	{
	  enum fp_obj new_objects[count - 1];
	  int j;
	  int k = 0;

	  for (j = 0 ; j < count ; j++)
	    {
	      if (j != i)
		new_objects[k++] = objects[j];
	    }

	  if (!test_permutations (new_objects, count - 1))
	    return false;
	}

      if (!unload_object (objects[i]))
	return false;
    }
  return true;
}

int
do_test (void)
{
#if HAVE_PRCTL_FP_MODE
  /* Determine available hardware support and current mode.  */
  current_fp_mode = prctl (PR_GET_FP_MODE);

  /* If the prctl call fails assume the core only has FR0 mode support.  */
  if (current_fp_mode == -1)
    current_fp_mode = 0;
  else
    {
      if (prctl (PR_SET_FP_MODE, 0) != 0)
	{
	  if (errno == ENOTSUP)
	    is_r6 = true;
	  else
	    {
	      printf ("unexpected error from PR_SET_FP_MODE, 0: %m\n");
	      return 1;
	    }
	}

      if (prctl (PR_SET_FP_MODE, PR_FP_MODE_FR) != 0)
	{
	  if (errno != ENOTSUP)
	    {
	      printf ("unexpected error from PR_SET_FP_MODE, "
		      "PR_FP_MODE_FR: %m\n");
	      return 1;
	    }
	}
      else
	has_fr1 = true;

      if (prctl (PR_SET_FP_MODE, PR_FP_MODE_FR | PR_FP_MODE_FRE) != 0)
	{
	  if (errno != ENOTSUP)
	    {
	      printf ("unexpected error from PR_SET_FP_MODE, "
		      "PR_FP_MODE_FR | PR_FP_MODE_FRE: %m\n");
	      return 1;
	    }
	}
      else
	has_fre = true;

      if (prctl (PR_SET_FP_MODE, current_fp_mode) != 0)
	{
	  printf ("unable to restore initial FP mode: %m\n");
	  return 1;
	}
    }

  if ((is_r6 && !(current_fp_mode & PR_FP_MODE_FR))
      || (!has_fr1 && (current_fp_mode & PR_FP_MODE_FR))
      || (!has_fre && (current_fp_mode & PR_FP_MODE_FRE)))
    {
      puts ("Inconsistency detected between initial FP mode "
	    "and supported FP modes\n");
      return 1;
    }
#else
  current_fp_mode = 0;
#endif

  /* Set up the initial state from executable and LDSO.  Assumptions:
     1) All system libraries have the same ABI as ld.so.
     2) Due to the fact that ld.so is tested by invoking it directly
        rather than via an interpreter, there is no point in varying
	the ABI of the test program.  Instead the ABI only varies for
	the shared libraries which get loaded.  */
  if (!set_next_fp_state (FPABI_NATIVE))
    {
      puts ("Unable to enter initial ABI state\n");
      return 1;
    }

  /* Compare the computed state with the hardware state.  */
  if (!mode_transition_valid_p ())
    return 1;

  /* Run all possible test permutations.  */
  if (!test_permutations (test_objects, FPABI_COUNT))
    {
      puts ("Mode checks failed\n");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../../test-skeleton.c"
