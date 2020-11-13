/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

#if defined(ABT_CONFIG_USE_MACH_ABSOLUTE_TIME)
static double g_time_mult = 0.0;
#endif
void ABTD_time_init(void)
{
#if defined(ABT_CONFIG_USE_MACH_ABSOLUTE_TIME)
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    g_time_mult = 1.0e-9 * ((double)info.numer / (double)info.denom);
#endif
}

/* Obtain the time value */
void ABTD_time_get(ABTD_time *p_time)
{
#if defined(ABT_CONFIG_USE_CLOCK_GETTIME)
    int ret = clock_gettime(CLOCK_REALTIME, p_time);
    ABTI_ASSERT(ret == 0);
#elif defined(ABT_CONFIG_USE_MACH_ABSOLUTE_TIME)
    *p_time = mach_absolute_time();
#elif defined(ABT_CONFIG_USE_GETTIMEOFDAY)
    int ret = gettimeofday(p_time, NULL);
    ABTI_ASSERT(ret == 0);
#endif
}

/* Read the time value as seconds (double precision) */
double ABTD_time_read_sec(ABTD_time *p_time)
{
    double secs;

#if defined(ABT_CONFIG_USE_CLOCK_GETTIME)
    secs = ((double)p_time->tv_sec) + 1.0e-9 * ((double)p_time->tv_nsec);
#elif defined(ABT_CONFIG_USE_MACH_ABSOLUTE_TIME)
    if (g_time_mult == 0.0)
        ABTD_time_init();
    secs = *p_time * g_time_mult;
#elif defined(ABT_CONFIG_USE_GETTIMEOFDAY)
    secs = ((double)p_time->tv_sec) + 1.0e-6 * ((double)p_time->tv_usec);
#endif

    return secs;
}
