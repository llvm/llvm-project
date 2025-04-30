/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#if defined(_WIN64)
#include "omp.h"

extern void omp_set_num_threads_(int num_threads)
{
  return omp_set_num_threads(num_threads);
}

extern int omp_get_num_threads_()
{
  return omp_get_num_threads();
}

extern int omp_get_max_threads_()
{
   return omp_get_max_threads();
}

extern int omp_get_thread_num_()
{
  return omp_get_thread_num();
}

extern int omp_get_num_procs_()
{
  return omp_get_num_procs();
}

extern int omp_in_parallel_()
{
  return omp_in_parallel();
}

extern void omp_set_dynamic_(int dynamic_threads)
{
  omp_set_dynamic(dynamic_threads);
}

extern int omp_get_dynamic_()
{
  return omp_get_dynamic();
}

extern void omp_set_nested_(int nested)
{
  omp_set_nested(nested);
}

extern int omp_get_nested_()
{
  return omp_get_nested();
}

extern double omp_get_wtime_()
{
  return omp_get_wtime();
}

extern double omp_get_wtick_()
{
  return omp_get_wtick();
}

extern int omp_get_thread_limit_()
{
  return omp_get_thread_limit();
}

extern void omp_set_max_active_levels_(int max_levels)
{
  omp_set_max_active_levels(max_levels);
}

extern int omp_get_max_active_levels_()
{
  return omp_get_max_active_levels();
}

extern int omp_get_level_()
{
  return omp_get_level();
}

extern int omp_get_ancestor_thread_num_(int level)
{
  return omp_get_ancestor_thread_num(level);
}

extern int omp_get_team_size_(int level)
{
  return omp_get_team_size(level);
}

extern int omp_get_active_level_()
{
  return omp_get_active_level();
}

extern int omp_in_final_()
{
  return omp_in_final();
}

extern int omp_get_cancellation_()
{
  return omp_get_cancellation();
}

extern int omp_get_num_places_()
{
  return omp_get_num_places();
}

extern int omp_get_place_num_()
{
  return omp_get_place_num();
}

extern int omp_get_partition_num_places_()
{
  return omp_get_partition_num_places();
}

extern void omp_set_default_device_(int device_num)
{
  omp_set_default_device(device_num);
}

extern int omp_get_default_device_()
{
  return omp_get_default_device();
}

extern int omp_get_num_teams_()
{
  return omp_get_num_teams();
}

extern int omp_get_team_num_()
{
  return omp_get_team_num();
}

extern int omp_is_initial_device_()
{
  return omp_is_initial_device();
}

extern int omp_get_max_task_priority_()
{
  return omp_get_max_task_priority();
}
#endif
