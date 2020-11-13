/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/* The purpose of this test is to check that ABT_sched_config_create,
 * ABT_sched_config_read, and ABT_sched_config_free behave as we want.
 */

#include <stdarg.h>
#include "abt.h"
#include "abttest.h"

ABT_sched_config_var param_a = { .idx = 0, .type = ABT_SCHED_CONFIG_INT };

ABT_sched_config_var param_b = { .idx = 1, .type = ABT_SCHED_CONFIG_DOUBLE };

int main(int argc, char *argv[])
{
    int a = 5;
    double b = 3.0;
    ABT_sched_config config1, config2, config3, config4;

    int a2;
    double b2;

    /* Initialize */
    ATS_read_args(argc, argv);
    ATS_init(argc, argv, 1);

    ABT_sched_config_create(&config1, param_a, a, ABT_sched_config_var_end);
    a2 = 0;
    b2 = 0.0;
    ABT_sched_config_read(config1, 2, &a2, &b2);
    ABT_sched_config_free(&config1);
    assert(a2 == a && b2 == 0.0);

    ABT_sched_config_create(&config2, param_b, b, ABT_sched_config_var_end);
    a2 = 0;
    b2 = 0.0;
    ABT_sched_config_read(config2, 2, &a2, &b2);
    ABT_sched_config_free(&config2);
    assert(a2 == 0 && b2 == b);

    ABT_sched_config_create(&config3, param_a, a, param_b, b,
                            ABT_sched_config_var_end);
    a2 = 0;
    b2 = 0.0;
    ABT_sched_config_read(config3, 2, &a2, &b2);
    ABT_sched_config_free(&config3);
    assert(a2 == a && b2 == b);

    ABT_sched_config_create(&config4, param_b, b, param_a, a,
                            ABT_sched_config_var_end);
    a2 = 0;
    b2 = 0.0;
    ABT_sched_config_read(config4, 2, &a2, &b2);
    ABT_sched_config_free(&config4);
    assert(a2 == a && b2 == b);

    /* Finalize */
    return ATS_finalize(0);
}
