/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef BENCH_UTIL_H_INCLUDED
#define BENCH_UTIL_H_INCLUDED

#ifdef USE_PAPI
#define ABTX_papi_assert(expr)                                                 \
    do {                                                                       \
        if (expr != PAPI_OK) {                                                 \
            fprintf(stderr, "Error at " #expr "\n");                           \
            exit(-1);                                                          \
        }                                                                      \
    } while (0)
#endif

#ifndef USE_PAPI
#define ABTX_start_prof(start_time, evset)                                     \
    do {                                                                       \
        start_time = ATS_get_cycles();                                         \
    } while (0)
#else
#define ABTX_start_prof(start_time, evset)                                     \
    do {                                                                       \
        start_time = ATS_get_cycles();                                         \
        ABTX_papi_assert(PAPI_start(evset));                                   \
    } while (0)
#endif

#ifndef USE_PAPI
#define ABTX_stop_prof(start_time, num, time_sum, time_sqrsum, evset, vals,    \
                       llcm_sum, llcm_sqrsum, totm_sum, tlbm_sqrsum)           \
    do {                                                                       \
        float elaps_time = (float)(ATS_get_cycles() - start_time) / num;       \
        time_sum += elaps_time;                                                \
        time_sqrsum += elaps_time * elaps_time;                                \
    } while (0)
#else
#define ABTX_stop_prof(start_time, num, time_sum, time_sqrsum, evset, vals,    \
                       llcm_sum, llcm_sqrsum, totm_sum, tlbm_sqrsum)           \
    do {                                                                       \
        ABTX_papi_assert(PAPI_stop(evset, vals));                              \
        float llcm = (float)vals[0] / num, tlbm = (float)vals[1] / num;        \
        llcm_sum += llcm;                                                      \
        llcm_sqrsum += llcm * llcm;                                            \
        totm_sum += tlbm;                                                      \
        tlbm_sqrsum += tlbm * tlbm;                                            \
        float elaps_time = (float)(ATS_get_cycles() - start_time) / num;       \
        time_sum += elaps_time;                                                \
        time_sqrsum += elaps_time * elaps_time;                                \
    } while (0)
#endif

#ifdef USE_PAPI
static inline void ABTX_papi_add_event(int event_set)
{
    int event_code[2];

#if (defined __MIC__) || (defined __KNC__)
#ifdef USE_PAPI_L1M_L2M
    event_code[0] = PAPI_L1_DCM;
    event_code[1] = PAPI_L1_ICM;
#else
    event_code[0] = PAPI_L2_LDM;
    event_code[1] = PAPI_TLB_DM;
#endif /* USE_PAPI_L1M_L2M */
#else
#ifdef USE_PAPI_L1M_L2M
    event_code[0] = PAPI_L1_TCM;
    event_code[1] = PAPI_L2_TCM;
#else
    event_code[0] = PAPI_L3_TCM;
    event_code[1] = PAPI_TLB_DM;
#endif /* USE_PAPI_L1M_L2M */
#endif

    ABTX_papi_assert(PAPI_add_event(event_set, event_code[0]));
    ABTX_papi_assert(PAPI_add_event(event_set, event_code[1]));
}
#endif

static inline void print_header(char *wu, int need_join)
{
    int line_size;

#ifndef USE_PAPI
    line_size = need_join ? 86 : 65;
    ATS_print_line(stdout, '-', line_size);
    printf("%-3s %8s %8s %22s ", "ES#", wu, "#Iter", "Create: cycles [std]");
    if (need_join)
        printf("%20s ", "Join: cycles [std]");
    printf("%20s\n", "Free: cycles [std]");
#else

    line_size = need_join ? 176 : 125;
    ATS_print_line(stdout, '-', line_size);
    printf("%-3s %8s %8s ", "ES#", wu, "#Iter");
#if (defined __MIC__) || (defined __KNC__)
#ifdef USE_PAPI_L1M_L2M
    printf("%22s %14s %14s ", "Create: cycles [std]", "L1Dm [std]",
           "L1Im [std]");
    if (need_join)
        printf("%20s %14s %14s ", "Join: cycles [std]", "L1Dm [std]",
               "L1Im [std]");
    printf("%20s %14s %14s\n", "Free: cycles [std]", "L1Dm [std]",
           "L1Im [std]");
#else
    printf("%22s %14s %14s ", "Create: cycles [std]", "L2Dm [std]",
           "TLBm [std]");
    if (need_join)
        printf("%20s %14s %14s ", "Join: cycles [std]", "L2Dm [std]",
               "TLBm [std]");
    printf("%20s %14s %14s\n", "Free: cycles [std]", "L2Dm [std]",
           "TLBm [std]");
#endif /* USE_PAPI_L1M_L2M */
#else
#ifdef USE_PAPI_L1M_L2M
    printf("%22s %14s %14s ", "Create: cycles [std]", "L1Cm [std]",
           "L2Cm [std]");
    if (need_join)
        printf("%20s %14s %14s", "Join: cycles [std]", "L1Cm [std]",
               "L2Cm [std]");
    printf("%20s %14s %14s\n", "Free: cycles [std]", "L1Cm [std]",
           "L2Cm [std]");
#else
    printf("%22s %14s %14s ", "Create: cycles [std]", "LLCm [std]",
           "TLBm [std]");
    if (need_join)
        printf("%20s %14s %14s ", "Join: cycles [std]", "LLCm [std]",
               "TLBm [std]");
    printf("%20s %14s %14s\n", "Free: cycles [std]", "LLCm [std]",
           "TLBm [std]");
#endif /* USE_PAPI_L1M_L2M */
#endif
#endif /* USE_PAPI */

    ATS_print_line(stdout, '-', line_size);
}

#ifdef USE_PAPI
static inline unsigned long ABTX_xstream_get_self(void)
{
    ABT_xstream self;
    ABT_xstream_self(&self);
    return (unsigned long)self;
}
#endif

/* The goal of the following sequence generator is to output data points
 * spread uniformly on a logarithmic scale. For example, if one wants to
 * generate powers two to be used as x-axis labels, but the number of ticks is
 * too small, this generator can output more terms to be used as labels while
 * having a uniform distribution on a log_2 scale. */

/* The following data-structure holds the state of the sequence generator */
typedef struct seq_state_t {
    int base;
    int prev_term;     /* previously generated term */
    int cur_stride;    /* current stride */
    int last_pow_term; /* last term which is power of base */
    /* maximum number of terms non-power of base between two
     * successive powers of base */
    int max_nonpow_terms;
} seq_state_t;

static inline void seq_init(seq_state_t *state, const int base,
                            const int prev_term, const int last_pow_term,
                            const int max_nonpow_terms)
{
    state->base = base;
    state->prev_term = prev_term;
    state->cur_stride = prev_term;
    state->last_pow_term = last_pow_term;
    state->max_nonpow_terms = max_nonpow_terms;
}

/* Core of the sequence generator */
static inline int seq_get_next_term(seq_state_t *state)
{
    int cur_term; /* term to return */
    cur_term = state->prev_term + state->cur_stride;
    if (cur_term == state->last_pow_term * state->base) {
        while (cur_term / state->cur_stride - 1 > state->max_nonpow_terms)
            state->cur_stride *= state->base;
        state->last_pow_term = cur_term;
    }
    state->prev_term = cur_term;
    return cur_term;
}

#endif /* BENCH_UTIL_H_INCLUDED */
