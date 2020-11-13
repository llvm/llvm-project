/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <pthread.h>
#include "abt.h"
#include "abttest.h"

static int g_verbose = 0;
static int g_num_errs = 0;

/* NOTE: The below NUM_ARG_KINDS should match the number of values in enum
 * ATS_arg in abttest.h. */
#define NUM_ARG_KINDS 4
static int g_arg_val[NUM_ARG_KINDS];

static void ATS_tool_init();
static void ATS_tool_finialize();
static ABT_bool g_tool_enabled;

void ATS_init(int argc, char **argv, int num_xstreams)
{
    int ret;
    char *envval;

    /* ABT_MAX_NUM_XSTREAMS determines the size of internal ES array */
    char snprintf_buffer[128];
    sprintf(snprintf_buffer, "ABT_MAX_NUM_XSTREAMS=%d", num_xstreams);
    putenv(snprintf_buffer);

    /* Initialize Argobots */
    ret = ABT_init(argc, argv);
    ATS_ERROR(ret, "ABT_init");

    /* Check environment variables */
    envval = getenv("ATS_VERBOSE");
    if (envval) {
        char *endptr;
        long val = strtol(envval, &endptr, 0);
        if (endptr == envval) {
            /* No digits are found */
            fprintf(stderr, "[Warning] %s is invalid for ATS_VERBOSE\n",
                    envval);
            fflush(stderr);
        } else if (val >= 0) {
            g_verbose = val;
        } else {
            /* Negative value */
            fprintf(stderr, "WARNING: %s is invalid for ATS_VERBOSE\n", envval);
            fflush(stderr);
        }
    }

    ret = ABT_info_query_config(ABT_INFO_QUERY_KIND_ENABLED_TOOL,
                                &g_tool_enabled);
    ATS_ERROR(ret, "ABT_info_query_config");
    envval = getenv("ATS_ENABLE_TOOL");
    if (envval && atoi(envval) == 0) {
        g_tool_enabled = 0;
    }

    if (g_tool_enabled) {
        /* Let's debug the tool interface as well. */
        ATS_tool_init();
    }
}

int ATS_finalize(int err)
{
    int ret;

    if (g_tool_enabled) {
        /* Finalize the tool interface. */
        ATS_tool_finialize();
    }

    /* Finalize Argobots */
    ret = ABT_finalize();
    ATS_ERROR(ret, "ABT_finalize");

    if (g_num_errs > 0) {
        printf("Found %d errors\n", g_num_errs);
        ret = EXIT_FAILURE;
    } else if (err != 0) {
        printf("ERROR: code=%d\n", err);
        ret = EXIT_FAILURE;
    } else {
        printf("No Errors\n");
        ret = EXIT_SUCCESS;
    }
    fflush(stdout);

    return ret;
}

void ATS_printf(int level, const char *format, ...)
{
    va_list list;

    if (g_verbose && level <= g_verbose) {
        va_start(list, format);
        vprintf(format, list);
        va_end(list);
        fflush(stdout);
    }
}

void ATS_error(int err, const char *msg, const char *file, int line)
{
    char *err_str;
    size_t len;
    int ret;

    if (err == ABT_SUCCESS)
        return;
    if (err == ABT_ERR_FEATURE_NA) {
        printf("Skipped\n");
        fflush(stdout);
        exit(77);
    }

    ret = ABT_error_get_str(err, NULL, &len);
    assert(ret == ABT_SUCCESS);
    err_str = (char *)malloc(sizeof(char) * len + 1);
    assert(err_str != NULL);
    ret = ABT_error_get_str(err, err_str, NULL);

    fprintf(stderr, "%s (%d): %s (%s:%d)\n", err_str, err, msg, file, line);

    free(err_str);

    g_num_errs++;

    exit(EXIT_FAILURE);
}

void ATS_error_if(int cond, const char *msg, const char *file, int line)
{
    if (!cond)
        return;
    fprintf(stderr, "%s (%s:%d)\n", msg, file, line);

    g_num_errs++;

    exit(EXIT_FAILURE);
}

static void ATS_print_help(char *prog)
{
    fprintf(stderr,
            "Usage: %s [-e num_es] [-u num_ult] [-t num_task] "
            "[-i iter] [-v verbose_level]\n",
            prog);
    fflush(stderr);
}

void ATS_read_args(int argc, char **argv)
{
    static int read = 0;
    int i, opt;

    if (read == 0)
        read = 1;
    else
        return;

    for (i = 0; i < NUM_ARG_KINDS; i++) {
        g_arg_val[i] = 1;
    }

    opterr = 0;
    while ((opt = getopt(argc, argv, "he:u:t:i:v:")) != -1) {
        switch (opt) {
            case 'e':
                g_arg_val[ATS_ARG_N_ES] = atoi(optarg);
                break;
            case 'u':
                g_arg_val[ATS_ARG_N_ULT] = atoi(optarg);
                break;
            case 't':
                g_arg_val[ATS_ARG_N_TASK] = atoi(optarg);
                break;
            case 'i':
                g_arg_val[ATS_ARG_N_ITER] = atoi(optarg);
                break;
            case 'v':
                g_verbose = atoi(optarg);
                break;
            case 'h':
                ATS_print_help(argv[0]);
                exit(EXIT_SUCCESS);
        }
    }
}

int ATS_get_arg_val(ATS_arg arg)
{
    if (arg < ATS_ARG_N_ES || (int)arg >= NUM_ARG_KINDS) {
        return 0;
    }
    return g_arg_val[arg];
}

void ATS_print_line(FILE *fp, char c, int len)
{
    int i;
    for (i = 0; i < len; i++) {
        fprintf(fp, "%c", c);
    }
    fprintf(fp, "\n");
    fflush(fp);
}

typedef enum {
    ATS_TOOL_UNIT_STATE_UNINIT = 0,
    ATS_TOOL_UNIT_STATE_READY,
    ATS_TOOL_UNIT_STATE_RUNNING,
    ATS_TOOL_UNIT_STATE_BLOCKED,
    ATS_TOOL_UNIT_STATE_FINISHED,
    ATS_TOOL_UNIT_STATE_JOINED,
} ATS_tool_unit_state;

typedef struct ATS_tool_unit_entry {
    const void *unit;
    ATS_tool_unit_state state;
    ABT_xstream last_xstream;
    struct ATS_tool_unit_entry *p_next;
} ATS_tool_unit_entry;

/* ABT_tool_unit_entry_table_index() assumes the following constant is 256. */
#define ATS_TOOL_UNIT_ENTRY_TABLE_NUM_ENTIRES 256
typedef struct {
    /* The simplest hast table with a lock.  Since we cannot use Argobots
     * locks in a callback handler, the following uses pthread_mutex. */
    pthread_mutex_t lock;
    ATS_tool_unit_entry *entries[ATS_TOOL_UNIT_ENTRY_TABLE_NUM_ENTIRES];
} ATS_tool_unit_entry_table;
static ATS_tool_unit_entry_table g_tool_unit_entry_table;

static inline size_t ABT_tool_unit_entry_table_index(const void *unit)
{
    /* Xor the pointer value. */
    if (sizeof(void *) == 4) {
        uint32_t val = (uint32_t)(uintptr_t)unit;
        uint32_t val2 = val ^ (val >> 16);
        return (val2 ^ (val2 >> 8)) &
               (ATS_TOOL_UNIT_ENTRY_TABLE_NUM_ENTIRES - 1);
    } else if (sizeof(void *) == 8) {
        uint64_t val = (uint64_t)(uintptr_t)unit;
        uint64_t val2 = val ^ (val >> 32);
        uint64_t val3 = val2 ^ (val2 >> 16);
        return (val3 ^ (val3 >> 8)) &
               (ATS_TOOL_UNIT_ENTRY_TABLE_NUM_ENTIRES - 1);
    }
    return 0;
}

static ATS_tool_unit_entry *ATS_tool_get_unit_entry(const void *unit)
{
    ATS_tool_unit_entry_table *p_table = &g_tool_unit_entry_table;
    ATS_tool_unit_entry *p_ret = NULL;
    pthread_mutex_lock(&p_table->lock);
    size_t index = ABT_tool_unit_entry_table_index(unit);
    ATS_tool_unit_entry *p_cur = p_table->entries[index];
    if (!p_cur) {
        p_ret = (ATS_tool_unit_entry *)calloc(1, sizeof(ATS_tool_unit_entry));
        p_ret->unit = unit;
        p_table->entries[index] = p_ret;
    } else {
        do {
            if (p_cur->unit == unit) {
                /* Already created. */
                p_ret = p_cur;
                break;
            } else if (!p_cur->p_next) {
                p_ret =
                    (ATS_tool_unit_entry *)calloc(1,
                                                  sizeof(ATS_tool_unit_entry));
                p_ret->unit = unit;
                p_cur->p_next = p_ret;
                break;
            }
            p_cur = p_cur->p_next;
        } while (1);
    }
    pthread_mutex_unlock(&p_table->lock);
    return p_ret;
}

static void ATS_tool_remove_unit_entry(const void *unit)
{
    ATS_tool_unit_entry_table *p_table = &g_tool_unit_entry_table;
    pthread_mutex_lock(&g_tool_unit_entry_table.lock);
    size_t index = ABT_tool_unit_entry_table_index(unit);
    ATS_tool_unit_entry *p_cur = p_table->entries[index];
    if (p_cur == NULL) {
        /* Not registered / double free */
        ATS_ERROR(ABT_ERR_OTHER, "ATS_tool_remove_unit_entry");
    } else if (p_cur->unit == unit) {
        p_table->entries[index] = p_cur->p_next;
        free(p_cur);
    } else {
        while (1) {
            ATS_tool_unit_entry *p_next = p_cur->p_next;
            if (!p_next) {
                /* Not registered / double free */
                ATS_ERROR(ABT_ERR_OTHER, "ATS_tool_remove_unit_entry");
            }
            if (p_next->unit == unit) {
                p_cur->p_next = p_next->p_next;
                free(p_next);
                break;
            }
            p_cur = p_next;
        }
    }
    pthread_mutex_unlock(&g_tool_unit_entry_table.lock);
}

static void ATS_tool_thread_callback(ABT_thread thread, ABT_xstream xstream,
                                     uint64_t event, ABT_tool_context context,
                                     void *user_arg)
{
    ATS_tool_unit_entry *p_entry = ATS_tool_get_unit_entry((void *)thread);

    ABT_bool is_unnamed = ABT_FALSE;
    int ret = ABT_thread_is_unnamed(thread, &is_unnamed);
    ATS_ERROR(ret, "ABT_thread_is_unnamed");

    /* The main scheduler has been already running, but the state is
     * UNINIT since there is no chance to update it. */
    switch (event) {
        case ABT_TOOL_EVENT_THREAD_CREATE:
            ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_UNINIT);
            p_entry->state = ATS_TOOL_UNIT_STATE_READY;
            break;
        case ABT_TOOL_EVENT_THREAD_JOIN:
            ATS_ERROR_IF(is_unnamed ||
                         (p_entry->state != ATS_TOOL_UNIT_STATE_FINISHED &&
                          p_entry->state != ATS_TOOL_UNIT_STATE_JOINED));
            p_entry->state = ATS_TOOL_UNIT_STATE_JOINED;
            break;
        case ABT_TOOL_EVENT_THREAD_FREE:
            if (is_unnamed) {
                ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_UNINIT &&
                             p_entry->state != ATS_TOOL_UNIT_STATE_READY &&
                             p_entry->state != ATS_TOOL_UNIT_STATE_FINISHED);
            } else {
                ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_UNINIT &&
                             p_entry->state != ATS_TOOL_UNIT_STATE_JOINED);
            }
            ATS_tool_remove_unit_entry((void *)thread);
            break;
        case ABT_TOOL_EVENT_THREAD_REVIVE:
            ATS_ERROR_IF(is_unnamed ||
                         p_entry->state != ATS_TOOL_UNIT_STATE_JOINED);
            p_entry->state = ATS_TOOL_UNIT_STATE_READY;
            break;
        case ABT_TOOL_EVENT_THREAD_RUN:
            ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_READY);
            p_entry->state = ATS_TOOL_UNIT_STATE_RUNNING;
            p_entry->last_xstream = xstream;
            break;
        case ABT_TOOL_EVENT_THREAD_FINISH:
            ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_UNINIT &&
                         !(p_entry->state == ATS_TOOL_UNIT_STATE_RUNNING &&
                           p_entry->last_xstream == xstream));
            p_entry->state = ATS_TOOL_UNIT_STATE_FINISHED;
            break;
        case ABT_TOOL_EVENT_THREAD_CANCEL:
            ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_READY);
            p_entry->state = ATS_TOOL_UNIT_STATE_FINISHED;
            break;
        case ABT_TOOL_EVENT_THREAD_YIELD:
            ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_UNINIT &&
                         !(p_entry->state == ATS_TOOL_UNIT_STATE_RUNNING &&
                           p_entry->last_xstream == xstream));
            p_entry->state = ATS_TOOL_UNIT_STATE_READY;
            break;
        case ABT_TOOL_EVENT_THREAD_SUSPEND:
            ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_UNINIT &&
                         !(p_entry->state == ATS_TOOL_UNIT_STATE_RUNNING &&
                           p_entry->last_xstream == xstream));
            p_entry->state = ATS_TOOL_UNIT_STATE_BLOCKED;
            break;
        case ABT_TOOL_EVENT_THREAD_RESUME:
            ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_BLOCKED);
            p_entry->state = ATS_TOOL_UNIT_STATE_READY;
            break;
        default:
            /* Unknown event. */
            ATS_ERROR(ABT_ERR_OTHER, "ATS_tool_thread_callback");
    }
}

static void ATS_tool_task_callback(ABT_task task, ABT_xstream xstream,
                                   uint64_t event, ABT_tool_context context,
                                   void *user_arg)
{
    ATS_tool_unit_entry *p_entry = ATS_tool_get_unit_entry((void *)task);

    ABT_bool is_unnamed = ABT_FALSE;
    int ret = ABT_task_is_unnamed(task, &is_unnamed);
    ATS_ERROR(ret, "ABT_task_is_unnamed");

    switch (event) {
        case ABT_TOOL_EVENT_TASK_CREATE:
            ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_UNINIT);
            p_entry->state = ATS_TOOL_UNIT_STATE_READY;
            break;
        case ABT_TOOL_EVENT_TASK_JOIN:
            ATS_ERROR_IF(is_unnamed ||
                         (p_entry->state != ATS_TOOL_UNIT_STATE_FINISHED &&
                          p_entry->state != ATS_TOOL_UNIT_STATE_JOINED));
            p_entry->state = ATS_TOOL_UNIT_STATE_JOINED;
            break;
        case ABT_TOOL_EVENT_TASK_FREE:
            /* The state can be ready if the created work unit cannot be pushed
             * to the pool. */
            if (is_unnamed) {
                ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_READY &&
                             p_entry->state != ATS_TOOL_UNIT_STATE_FINISHED);
            } else {
                ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_READY &&
                             p_entry->state != ATS_TOOL_UNIT_STATE_JOINED);
            }
            ATS_tool_remove_unit_entry((void *)task);
            break;
        case ABT_TOOL_EVENT_TASK_REVIVE:
            ATS_ERROR_IF(is_unnamed ||
                         p_entry->state != ATS_TOOL_UNIT_STATE_JOINED);
            p_entry->state = ATS_TOOL_UNIT_STATE_READY;
            break;
        case ABT_TOOL_EVENT_TASK_RUN:
            ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_READY);
            p_entry->state = ATS_TOOL_UNIT_STATE_RUNNING;
            p_entry->last_xstream = xstream;
            break;
        case ABT_TOOL_EVENT_TASK_FINISH:
            ATS_ERROR_IF(p_entry->state == ATS_TOOL_UNIT_STATE_RUNNING &&
                         p_entry->last_xstream != xstream);
            p_entry->state = ATS_TOOL_UNIT_STATE_FINISHED;
            break;
        case ABT_TOOL_EVENT_TASK_CANCEL:
            ATS_ERROR_IF(p_entry->state != ATS_TOOL_UNIT_STATE_READY);
            p_entry->state = ATS_TOOL_UNIT_STATE_FINISHED;
            break;
        default:
            /* Unknown event. */
            ATS_ERROR(ABT_ERR_OTHER, "ATS_tool_task_callback");
    }
}

static void ATS_tool_init()
{
    /* Initialize the hash table. */
    int ret, i;
    ATS_tool_unit_entry_table *p_table = &g_tool_unit_entry_table;
    pthread_mutex_init(&p_table->lock, NULL);
    for (i = 0; i < ATS_TOOL_UNIT_ENTRY_TABLE_NUM_ENTIRES; i++) {
        p_table->entries[i] = NULL;
    }
    /* Add this main thread. */
    ABT_thread self_thread;
    ABT_xstream self_xstream;
    ret = ABT_thread_self(&self_thread);
    ATS_ERROR(ret, "ABT_thread_self");
    ret = ABT_xstream_self(&self_xstream);
    ATS_ERROR(ret, "ABT_xstream_self");
    ATS_tool_unit_entry *p_entry = ATS_tool_get_unit_entry((void *)self_thread);
    p_entry->state = ATS_TOOL_UNIT_STATE_RUNNING;
    p_entry->last_xstream = self_xstream;

    ret = ABT_tool_register_thread_callback(ATS_tool_thread_callback,
                                            ABT_TOOL_EVENT_THREAD_ALL, NULL);
    ATS_ERROR(ret, "ABT_tool_register_thread_callback");
    ret = ABT_tool_register_task_callback(ATS_tool_task_callback,
                                          ABT_TOOL_EVENT_TASK_ALL, NULL);
    ATS_ERROR(ret, "ABT_tool_register_task_callback");
}

static void ATS_tool_finialize()
{
    int ret, i;
    ATS_tool_unit_entry_table *p_table = &g_tool_unit_entry_table;
    for (i = 0; i < ATS_TOOL_UNIT_ENTRY_TABLE_NUM_ENTIRES; i++) {
        ATS_tool_unit_entry *p_cur = p_table->entries[i];
        while (p_cur) {
            ATS_tool_unit_entry *p_next = p_cur->p_next;
            free(p_cur);
            p_cur = p_next;
        }
        p_table->entries[i] = NULL;
    }
    pthread_mutex_destroy(&p_table->lock);

    ret = ABT_tool_register_thread_callback(NULL, ABT_TOOL_EVENT_THREAD_NONE,
                                            NULL);
    ATS_ERROR(ret, "ABT_tool_register_thread_callback");
    ret = ABT_tool_register_task_callback(NULL, ABT_TOOL_EVENT_TASK_NONE, NULL);
    ATS_ERROR(ret, "ABT_tool_register_task_callback");
}
