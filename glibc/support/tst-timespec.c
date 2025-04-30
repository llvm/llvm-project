/* Test for support_timespec_check_in_range function.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <support/timespec.h>
#include <support/check.h>
#include <limits.h>
#include <intprops.h>

#define TIMESPEC_HZ 1000000000

struct timespec_ns_test_case
{
  struct timespec time;
  time_t time_ns;
};

struct timespec_norm_test_case
{
  struct timespec time;
  struct timespec norm;
};

struct timespec_test_case
{
  struct timespec expected;
  struct timespec observed;
  double upper_bound;
  double lower_bound;
  int result;
};

#define TIME_T_MIN TYPE_MINIMUM (time_t)
#define TIME_T_MAX TYPE_MAXIMUM (time_t)

/* Test cases for timespec_ns */
struct timespec_ns_test_case ns_cases[] = {
  {.time = {.tv_sec = 0, .tv_nsec = 0},
   .time_ns = 0,
  },
  {.time = {.tv_sec = 0, .tv_nsec = 1},
   .time_ns = 1,
  },
  {.time = {.tv_sec = 1, .tv_nsec = 0},
   .time_ns = TIMESPEC_HZ,
  },
  {.time = {.tv_sec = 1, .tv_nsec = 1},
   .time_ns = TIMESPEC_HZ + 1,
  },
  {.time = {.tv_sec = 0, .tv_nsec = -1},
   .time_ns = -1,
  },
  {.time = {.tv_sec = -1, .tv_nsec = 0},
   .time_ns = -TIMESPEC_HZ,
  },
  {.time = {.tv_sec = -1, .tv_nsec = -1},
   .time_ns = -TIMESPEC_HZ - 1,
  },
  {.time = {.tv_sec = 1, .tv_nsec = -1},
   .time_ns = TIMESPEC_HZ - 1,
  },
  {.time = {.tv_sec = -1, .tv_nsec = 1},
   .time_ns = -TIMESPEC_HZ + 1,
  },
  /* Overflow bondary by 2  */
  {.time = {.tv_sec = TIME_T_MAX / TIMESPEC_HZ,
	    .tv_nsec = TIME_T_MAX % TIMESPEC_HZ - 1},
   .time_ns = TIME_T_MAX - 1,
  },
  /* Overflow bondary  */
  {.time = {.tv_sec = TIME_T_MAX / TIMESPEC_HZ,
	    .tv_nsec = TIME_T_MAX % TIMESPEC_HZ},
   .time_ns = TIME_T_MAX,
  },
  /* Underflow bondary by 1  */
  {.time = {.tv_sec = TIME_T_MIN / TIMESPEC_HZ,
	    .tv_nsec = TIME_T_MIN % TIMESPEC_HZ + 1},
   .time_ns = TIME_T_MIN + 1,
  },
  /* Underflow bondary  */
  {.time = {.tv_sec = TIME_T_MIN / TIMESPEC_HZ,
	    .tv_nsec = TIME_T_MIN % TIMESPEC_HZ},
   .time_ns = TIME_T_MIN,
  },
  /* Multiplication overflow  */
  {.time = {.tv_sec = TIME_T_MAX / TIMESPEC_HZ + 1, .tv_nsec = 1},
   .time_ns = TIME_T_MAX,
  },
  /* Multiplication underflow  */
  {.time = {.tv_sec = TIME_T_MIN / TIMESPEC_HZ - 1, .tv_nsec = -1},
   .time_ns = TIME_T_MIN,
  },
  /* Sum overflows  */
  {.time = {.tv_sec = TIME_T_MAX / TIMESPEC_HZ,
	    .tv_nsec = TIME_T_MAX % TIMESPEC_HZ + 1},
   .time_ns = TIME_T_MAX,
  },
  /* Sum underflow  */
  {.time = {.tv_sec = TIME_T_MIN / TIMESPEC_HZ,
	    .tv_nsec = TIME_T_MIN % TIMESPEC_HZ - 1},
   .time_ns = TIME_T_MIN,
  }
};

/* Test cases for timespec_norm */
struct timespec_norm_test_case norm_cases[] = {
  /* Positive cases  */
  {.time = {.tv_sec = 0, .tv_nsec = 0},
   .norm = {.tv_sec = 0, .tv_nsec = 0}
  },
  {.time = {.tv_sec = 1, .tv_nsec = 0},
   .norm = {.tv_sec = 1, .tv_nsec = 0}
  },
  {.time = {.tv_sec = 0, .tv_nsec = 1},
   .norm = {.tv_sec = 0, .tv_nsec = 1}
  },
  {.time = {.tv_sec = 0, .tv_nsec = TIMESPEC_HZ},
   .norm = {.tv_sec = 1, .tv_nsec = 0}
  },
  {.time = {.tv_sec = 0, .tv_nsec = TIMESPEC_HZ + 1},
   .norm = {.tv_sec = 1, .tv_nsec = 1}
  },
  {.time = {.tv_sec = 1, .tv_nsec = TIMESPEC_HZ},
   .norm = {.tv_sec = 2, .tv_nsec = 0}
  },
  {.time = {.tv_sec = 1, .tv_nsec = TIMESPEC_HZ + 1},
   .norm = {.tv_sec = 2, .tv_nsec = 1}
  },
  /* Negative cases  */
  {.time = {.tv_sec = 0, .tv_nsec = -TIMESPEC_HZ},
   .norm = {.tv_sec = -1, .tv_nsec = 0}
  },
  {.time = {.tv_sec = 0, .tv_nsec = -TIMESPEC_HZ - 1},
   .norm = {.tv_sec = -1, .tv_nsec = -1}
  },
  {.time = {.tv_sec = -1, .tv_nsec = -TIMESPEC_HZ},
   .norm = {.tv_sec = -2, .tv_nsec = 0}
  },
  {.time = {.tv_sec = -1, .tv_nsec = -TIMESPEC_HZ - 1},
   .norm = {.tv_sec = -2, .tv_nsec = -1}
  },
  /* Overflow bondary by 2  */
  {.time = {.tv_sec = TIME_T_MAX - 2, .tv_nsec = TIMESPEC_HZ + 1},
   .norm = {.tv_sec = TIME_T_MAX - 1, 1},
  },
  /* Overflow bondary by 1  */
  {.time = {.tv_sec = TIME_T_MAX - 1, .tv_nsec = TIMESPEC_HZ + 1},
   .norm = {.tv_sec = TIME_T_MAX, .tv_nsec = 1},
  },
  /* Underflow bondary by 2  */
  {.time = {.tv_sec = TIME_T_MIN + 2, .tv_nsec = -TIMESPEC_HZ - 1},
   .norm = {.tv_sec = TIME_T_MIN + 1, -1},
  },
  /* Underflow bondary by 1  */
  {.time = {.tv_sec = TIME_T_MIN + 1, .tv_nsec = -TIMESPEC_HZ - 1},
   .norm = {.tv_sec = TIME_T_MIN, .tv_nsec = -1},
  },
  /* SUM overflow  */
  {.time = {.tv_sec = TIME_T_MAX, .tv_nsec = TIMESPEC_HZ},
   .norm = {.tv_sec = TIME_T_MAX, .tv_nsec = TIMESPEC_HZ - 1},
  },
  /* SUM underflow  */
  {.time = {.tv_sec = TIME_T_MIN, .tv_nsec = -TIMESPEC_HZ},
   .norm = {.tv_sec = TIME_T_MIN, .tv_nsec = -1 * (TIMESPEC_HZ - 1)},
  }
};

/* Test cases for timespec_check_in_range  */
struct timespec_test_case check_cases[] = {
  /* 0 - In range  */
  {.expected = {.tv_sec = 1, .tv_nsec = 0},
   .observed = {.tv_sec = 1, .tv_nsec = 0},
   .upper_bound = 1, .lower_bound = 1, .result = 1,
  },
  /* 1 - Out of range  */
  {.expected = {.tv_sec = 1, .tv_nsec = 0},
   .observed = {.tv_sec = 2, .tv_nsec = 0},
   .upper_bound = 1, .lower_bound = 1, .result = 0,
  },
  /* 2 - Upper Bound  */
  {.expected = {.tv_sec = 1, .tv_nsec = 0},
   .observed = {.tv_sec = 2, .tv_nsec = 0},
   .upper_bound = 2, .lower_bound = 1, .result = 1,
  },
  /* 3 - Lower Bound  */
  {.expected = {.tv_sec = 1, .tv_nsec = 0},
   .observed = {.tv_sec = 0, .tv_nsec = 0},
   .upper_bound = 1, .lower_bound = 0, .result = 1,
  },
  /* 4 - Out of range by nanosecs  */
  {.expected = {.tv_sec = 1, .tv_nsec = 0},
   .observed = {.tv_sec = 1, .tv_nsec = 500},
   .upper_bound = 1, .lower_bound = 1, .result = 0,
  },
  /* 5 - In range by nanosecs  */
  {.expected = {.tv_sec = 1, .tv_nsec = 0},
   .observed = {.tv_sec = 1, .tv_nsec = 50000},
   .upper_bound = 1.3, .lower_bound = 1, .result = 1,
  },
  /* 6 - Big nanosecs  */
  {.expected = {.tv_sec = 1, .tv_nsec = 0},
   .observed = {.tv_sec = 0, .tv_nsec = 4000000},
   .upper_bound = 1, .lower_bound = .001, .result = 1,
  },
  /* 7 - In range Negative values  */
  {.expected = {.tv_sec = -1, .tv_nsec = 0},
   .observed = {.tv_sec = -1, .tv_nsec = 0},
   .upper_bound = 1, .lower_bound = 1, .result = 1,
  },
  /* 8 - Out of range Negative values  */
  {.expected = {.tv_sec = -1, .tv_nsec = 0},
   .observed = {.tv_sec = -1, .tv_nsec = 0},
   .upper_bound = -1, .lower_bound = -1, .result = 0,
  },
  /* 9 - Negative values with negative nanosecs  */
  {.expected = {.tv_sec = -1, .tv_nsec = 0},
   .observed = {.tv_sec = -1, .tv_nsec = -2000},
   .upper_bound = 1, .lower_bound = 1, .result = 0,
  },
  /* 10 - Strict bounds  */
  {.expected = {.tv_sec = -1, .tv_nsec = 0},
   .observed = {.tv_sec = -1, .tv_nsec = -20000},
   .upper_bound = 1.00002, .lower_bound = 1.0000191, .result = 1,
  },
  /* 11 - Strict bounds with loose upper bound  */
  {.expected = {.tv_sec = 1, .tv_nsec = 20000},
   .observed = {.tv_sec = 1, .tv_nsec = 30000},
   .upper_bound = 1.0000100000, .lower_bound = 1.0000099998, .result = 1,
  },
  /* 12 - Strict bounds with loose lower bound  */
  {.expected = {.tv_sec = 1, .tv_nsec = 20000},
   .observed = {.tv_sec = 1, .tv_nsec = 30000},
   .upper_bound = 1.0000099999, .lower_bound = 1.00000999979, .result = 1,
  },
  /* 13 - Strict bounds highest precision  */
  {.expected = {.tv_sec = 1, .tv_nsec = 20000},
   .observed = {.tv_sec = 1, .tv_nsec = 30000},
   .upper_bound = 1.00000999980001, .lower_bound = 1.00000999979999, .result = 1,
  },
  /* Maximum/Minimum long values  */
  /* 14  */
  {.expected = {.tv_sec = TIME_T_MAX, .tv_nsec = TIMESPEC_HZ - 1},
   .observed = {.tv_sec = TIME_T_MAX, .tv_nsec = TIMESPEC_HZ - 2},
   .upper_bound = 1, .lower_bound = .9, .result = 1,
  },
  /* 15 - support_timespec_ns overflow  */
  {.expected = {.tv_sec = TIME_T_MAX, .tv_nsec = TIMESPEC_HZ},
   .observed = {.tv_sec = TIME_T_MAX, .tv_nsec = TIMESPEC_HZ},
   .upper_bound = 1, .lower_bound = 1, .result = 1,
  },
  /* 16 - support_timespec_ns overflow + underflow  */
  {.expected = {.tv_sec = TIME_T_MAX, .tv_nsec = TIMESPEC_HZ},
   .observed = {.tv_sec = TIME_T_MIN, .tv_nsec = -TIMESPEC_HZ},
   .upper_bound = 1, .lower_bound = 1, .result = 0,
  },
  /* 17 - support_timespec_ns underflow  */
  {.expected = {.tv_sec = TIME_T_MIN, .tv_nsec = -TIMESPEC_HZ},
   .observed = {.tv_sec = TIME_T_MIN, .tv_nsec = -TIMESPEC_HZ},
   .upper_bound = 1, .lower_bound = 1, .result = 1,
  },
  /* 18 - support_timespec_ns underflow + overflow  */
  {.expected = {.tv_sec = TIME_T_MIN, .tv_nsec = -TIMESPEC_HZ},
   .observed = {.tv_sec = TIME_T_MAX, .tv_nsec = TIMESPEC_HZ},
   .upper_bound = 1, .lower_bound = 1, .result = 0,
  },
  /* 19 - Biggest division  */
  {.expected = {.tv_sec = TIME_T_MAX / TIMESPEC_HZ,
		.tv_nsec = TIMESPEC_HZ - 1},
   .observed = {.tv_sec = 0, .tv_nsec = 1},
   .upper_bound = 1, .lower_bound = 1.0842021724855044e-19, .result = 1,
  },
  /* 20 - Lowest division  */
  {.expected = {.tv_sec = 0, .tv_nsec = 1},
   .observed = {.tv_sec = TIME_T_MAX / TIMESPEC_HZ,
		.tv_nsec = TIMESPEC_HZ - 1},
   .upper_bound = TIME_T_MAX, .lower_bound = 1, .result = 1,
  },
};

static int
do_test (void)
{
  int i = 0;
  int ntests = sizeof (ns_cases) / sizeof (ns_cases[0]);

  printf("Testing support_timespec_ns\n");
  for (i = 0; i < ntests; i++)
    {
      printf("Test case %d\n", i);
      TEST_COMPARE (support_timespec_ns (ns_cases[i].time),
		    ns_cases[i].time_ns);
    }

  ntests = sizeof (norm_cases) / sizeof (norm_cases[0]);
  struct timespec result;
  printf("Testing support_timespec_normalize\n");
  for (i = 0; i < ntests; i++)
    {
      printf("Test case %d\n", i);
      result = support_timespec_normalize (norm_cases[i].time);
      TEST_COMPARE (norm_cases[i].norm.tv_sec, result.tv_sec);
      TEST_COMPARE (norm_cases[i].norm.tv_nsec, result.tv_nsec);
    }

  ntests = sizeof (check_cases) / sizeof (check_cases[0]);
  printf("Testing support_timespec_check_in_range\n");
  for (i = 0; i < ntests; i++)
    {
      /* Its hard to find which test failed with just the TEST_COMPARE report.
         So here we print every running testcase as well.  */
      printf("Test case %d\n", i);
      TEST_COMPARE (support_timespec_check_in_range
		    (check_cases[i].expected, check_cases[i].observed,
		     check_cases[i].lower_bound,
		     check_cases[i].upper_bound), check_cases[i].result);
    }
  return 0;
}

#include <support/test-driver.c>
