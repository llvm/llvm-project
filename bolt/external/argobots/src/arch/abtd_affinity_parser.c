/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

static ABTD_affinity_id_list *id_list_create(void)
{
    ABTD_affinity_id_list *p_id_list;
    int ret =
        ABTU_calloc(1, sizeof(ABTD_affinity_id_list), (void **)&p_id_list);
    ABTI_ASSERT(ret == ABT_SUCCESS);
    return p_id_list;
}

static void id_list_free(ABTD_affinity_id_list *p_id_list)
{
    if (p_id_list)
        ABTU_free(p_id_list->ids);
    ABTU_free(p_id_list);
}

static void id_list_add(ABTD_affinity_id_list *p_id_list, int id, int num,
                        int stride)
{
    /* Needs to add num ids. */
    int i, ret;
    ret = ABTU_realloc(sizeof(int) * p_id_list->num,
                       sizeof(int) * (p_id_list->num + num),
                       (void **)&p_id_list->ids);
    ABTI_ASSERT(ret == ABT_SUCCESS);
    for (i = 0; i < num; i++) {
        p_id_list->ids[p_id_list->num + i] = id + stride * i;
    }
    p_id_list->num += num;
}

static ABTD_affinity_list *list_create(void)
{

    ABTD_affinity_list *p_list;
    int ret = ABTU_calloc(1, sizeof(ABTD_affinity_list), (void **)&p_list);
    ABTI_ASSERT(ret == ABT_SUCCESS);
    return p_list;
}

static void list_free(ABTD_affinity_list *p_list)
{
    if (p_list) {
        int i;
        for (i = 0; i < p_list->num; i++)
            id_list_free(p_list->p_id_lists[i]);
        free(p_list->p_id_lists);
    }
    free(p_list);
}

static void list_add(ABTD_affinity_list *p_list, ABTD_affinity_id_list *p_base,
                     int num, int stride)
{
    /* Needs to add num id-lists. */
    int i, j, ret;

    ret = ABTU_realloc(sizeof(ABTD_affinity_id_list *) * p_list->num,
                       sizeof(ABTD_affinity_id_list *) * (p_list->num + num),
                       (void **)&p_list->p_id_lists);
    ABTI_ASSERT(ret == ABT_SUCCESS);
    for (i = 1; i < num; i++) {
        ABTD_affinity_id_list *p_id_list = id_list_create();
        p_id_list->num = p_base->num;
        ret =
            ABTU_malloc(sizeof(int) * p_id_list->num, (void **)&p_id_list->ids);
        ABTI_ASSERT(ret == ABT_SUCCESS);
        for (j = 0; j < p_id_list->num; j++)
            p_id_list->ids[j] = p_base->ids[j] + stride * i;
        p_list->p_id_lists[p_list->num + i] = p_id_list;
    }
    p_list->p_id_lists[p_list->num] = p_base;
    p_list->num += num;
}

static inline int is_whitespace(char c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

/* Integer. */
static int consume_int(const char *str, int *p_index, int *p_val)
{
    int index = *p_index, val = 0, val_sign = 1;
    char flag = 'n';
    while (1) {
        char c = *(str + index);
        if (flag != 'v' && c == '-') {
            /* Negative sign. */
            flag = 's';
            val_sign = -val_sign;
        } else if (flag != 'v' && c == '+') {
            /* Positive sign. */
            flag = 's';
        } else if (flag == 'n' && is_whitespace(c)) {
            /* Skip a whitespace. */
        } else if ('0' <= c && c <= '9') {
            /* Value. */
            flag = 'v';
            val = val * 10 + (int)(c - '0');
        } else {
            /* Encounters a symbol. */
            if (flag == 'v') {
                /* Succeeded. */
                *p_val = val * val_sign;
                *p_index = index;
                return 1;
            } else {
                /* Failed. The parser could not consume a value. */
                return 0;
            }
        }
        index++;
    }
}

/* Positive integer */
static int consume_pint(const char *str, int *p_index, int *p_val)
{
    int index = *p_index, val;
    /* The value must be positive. */
    if (consume_int(str, &index, &val) && val > 0) {
        *p_index = index;
        *p_val = val;
        return 1;
    }
    return 0;
}

/* Symbol.  If succeeded, it returns a consumed characters. */
static int consume_symbol(const char *str, int *p_index, char symbol)
{
    int index = *p_index;
    while (1) {
        char c = *(str + index);
        if (c == symbol) {
            *p_index = index + 1;
            return 1;
        } else if (is_whitespace(c)) {
            /* Skip a whitespace. */
        } else {
            /* Failed. The parser could not consume a symbol. */
            return 0;
        }
        index++;
    }
}

static ABTD_affinity_id_list *parse_es_id_list(const char *affinity_str,
                                               int *p_index)
{
    ABTD_affinity_id_list *p_id_list = id_list_create();
    int val;
    /* Expect either <id> or { <id-list> } */
    if (consume_int(affinity_str, p_index, &val)) {
        /* If the first token is an integer, it is <id> */
        id_list_add(p_id_list, val, 1, 1);
        return p_id_list;
    } else if (consume_symbol(affinity_str, p_index, '{')) {
        /* It should be "{" <id-list> "}".  Parse <id-list> and "}" */
        while (1) {
            int id, num = 1, stride = 1;
            /* Parse <id-interval>.  First, expect <id> */
            if (!consume_int(affinity_str, p_index, &id))
                goto FAILED;
            /* Optional: ":" <num> */
            if (consume_symbol(affinity_str, p_index, ':')) {
                /* Expect <num> */
                if (!consume_pint(affinity_str, p_index, &num))
                    goto FAILED;
                /* Optional: ":" <stride> */
                if (consume_symbol(affinity_str, p_index, ':')) {
                    /* Expect <stride> */
                    if (!consume_int(affinity_str, p_index, &stride))
                        goto FAILED;
                }
            }
            /* Add ids based on <id-interval> */
            id_list_add(p_id_list, id, num, stride);
            /* After <id-interval>, we expect either "," (in <id-list>) or "}"
             * (in <es-id-list>) */
            if (consume_symbol(affinity_str, p_index, ',')) {
                /* Parse <id-interval> again. */
                continue;
            }
            /* Expect "}" */
            if (!consume_symbol(affinity_str, p_index, '}'))
                goto FAILED;
            /* Succeeded. */
            return p_id_list;
        }
    }
FAILED:
    id_list_free(p_id_list);
    return NULL; /* Failed. */
}

static ABTD_affinity_list *parse_list(const char *affinity_str)
{
    if (!affinity_str)
        return NULL;
    int index = 0;
    ABTD_affinity_list *p_list = list_create();
    ABTD_affinity_id_list *p_id_list = NULL;
    while (1) {
        int num = 1, stride = 1;
        /* Parse <interval> */
        /* Expect <es-id-list> */
        p_id_list = parse_es_id_list(affinity_str, &index);
        if (!p_id_list)
            goto FAILED;
        /* Optional: ":" <num> */
        if (consume_symbol(affinity_str, &index, ':')) {
            /* Expect <num> */
            if (!consume_pint(affinity_str, &index, &num))
                goto FAILED;
            /* Optional: ":" <stride> */
            if (consume_symbol(affinity_str, &index, ':')) {
                /* Expect <stride> */
                if (!consume_int(affinity_str, &index, &stride))
                    goto FAILED;
            }
        }
        /* Add <es-id-list> based on <interval> */
        list_add(p_list, p_id_list, num, stride);
        p_id_list = NULL;
        /* After <interval>, expect either "," (in <list>) or "\0" */
        if (consume_symbol(affinity_str, &index, ',')) {
            /* Parse <interval> again. */
            continue;
        }
        /* Expect "\0" */
        if (!consume_symbol(affinity_str, &index, '\0'))
            goto FAILED;
        /* Succeeded. */
        return p_list;
    }
FAILED:
    list_free(p_list);
    id_list_free(p_id_list);
    return NULL; /* Failed. */
}

ABTD_affinity_list *ABTD_affinity_list_create(const char *affinity_str)
{
    return parse_list(affinity_str);
}

void ABTD_affinity_list_free(ABTD_affinity_list *p_list)
{
    list_free(p_list);
}

#if 0

static int is_equal(const ABTD_affinity_list *a, const ABTD_affinity_list *b)
{
    int i, j;
    if (a->num != b->num)
        return 0;
    for (i = 0; i < a->num; i++) {
        const ABTD_affinity_id_list *a_id = a->p_id_lists[i];
        const ABTD_affinity_id_list *b_id = b->p_id_lists[i];
        if (a_id->num != b_id->num)
            return 0;
        for (j = 0; j < a_id->num; j++) {
            if (a_id->ids[j] != b_id->ids[j])
                return 0;
        }
    }
    return 1;
}

static int is_equal_str(const char *a_str, const char *b_str)
{
    int ret = 1;
    ABTD_affinity_list *a = parse_list(a_str);
    ABTD_affinity_list *b = parse_list(b_str);
    ret = a && b && is_equal(a, b);
    list_free(a);
    list_free(b);
    return ret;
}

static int is_err_str(const char *str)
{
    ABTD_affinity_list *a = parse_list(str);
    if (a) {
        list_free(a);
        return 0;
    }
    return 1;
}

static void test_parse(void)
{
    /* Legal strings */
    assert(!is_err_str("++1"));
    assert(!is_err_str("+-1"));
    assert(!is_err_str("+-+-1"));
    assert(!is_err_str("+0"));
    assert(!is_err_str("-0"));
    assert(!is_err_str("-9:1:-9"));
    assert(!is_err_str("-9:1:0"));
    assert(!is_err_str("-9:1:9"));
    assert(!is_err_str("0:1:-9"));
    assert(!is_err_str("0:1:0"));
    assert(!is_err_str("0:1:9"));
    assert(!is_err_str("9:1:-9"));
    assert(!is_err_str("9:1:0"));
    assert(!is_err_str("9:1:9"));
    assert(!is_err_str("{-9:1:-9}"));
    assert(!is_err_str("{-9:1:0}"));
    assert(!is_err_str("{-9:1:9}"));
    assert(!is_err_str("{0:1:-9}"));
    assert(!is_err_str("{0:1:0}"));
    assert(!is_err_str("{0:1:9}"));
    assert(!is_err_str("{9:1:-9}"));
    assert(!is_err_str("{9:1:0}"));
    assert(!is_err_str("{9:1:9}"));
    assert(!is_err_str("1,2,3"));
    assert(!is_err_str("1,2,{1,2}"));
    assert(!is_err_str("1,2,{1:2}"));
    assert(!is_err_str("1:2,{1:2}"));
    assert(!is_err_str("1:2:1,2"));
    assert(!is_err_str(" 1 :  +2 , { -1 : \r 2\n:2}\n"));
    /* Illegal strings */
    assert(is_err_str(""));
    assert(is_err_str("{}"));
    assert(is_err_str("+ 1"));
    assert(is_err_str("+ +1"));
    assert(is_err_str("+ -1"));
    assert(is_err_str("1:"));
    assert(is_err_str("1:2:"));
    assert(is_err_str("1:2,"));
    assert(is_err_str("1:-2"));
    assert(is_err_str("1:0"));
    assert(is_err_str("1:-2:4"));
    assert(is_err_str("1:0:4"));
    assert(is_err_str("1:1:1:"));
    assert(is_err_str("1:1:1:1"));
    assert(is_err_str("1:1:1:1,1"));
    assert(is_err_str("{1:2:3},"));
    assert(is_err_str("{1:2:3}:"));
    assert(is_err_str("{1:2:3}:2:"));
    assert(is_err_str("{:2:3}"));
    assert(is_err_str("{{2:3}}"));
    assert(is_err_str("{2:3}}"));
    assert(is_err_str("2:3}"));
    assert(is_err_str("{1:2:3"));
    assert(is_err_str("{1,2,}"));
    assert(is_err_str("{1:-2}"));
    assert(is_err_str("{1:0}"));
    assert(is_err_str("{1:-2:4}"));
    assert(is_err_str("{1:0:4}"));
    /* Comparison */
    assert(is_equal_str("{1},{2},{3},{4}", "1,2,3,4"));
    assert(is_equal_str("{1:4:1}", "{1,2,3,4}"));
    assert(is_equal_str("{1:4}", "{1,2,3,4}"));
    assert(is_equal_str("1:2,3:2", "1,2,3,4"));
    assert(is_equal_str("{1:2},3:2", "{1,2},3,4"));
    assert(is_equal_str("{1:1:4},{2:1:-4},{3:1:0},{4:1}", "1,2,3,4"));
    assert(is_equal_str("{3:4:-1}", "{3,2,1,0}"));
    assert(is_equal_str("3:4:-1,-1", "3,2,1,0,-1"));
    assert(is_equal_str("{1:2:3}:1", "{1,4}"));
    assert(is_equal_str("{1:2:3}:3", "{1,4},{2,5},{3,6}"));
    assert(is_equal_str("{1:2:3}:3:2", "{1,4},{3,6},{5,8}"));
    assert(is_equal_str("{1:2:3}:3:-2", "{1,4},{-1,2},{-3,0}"));
    assert(is_equal_str("{1:2:3}:3:-2,1", "{1,4},{-1,2},{-3,0},1"));
    assert(is_equal_str("{-2:3:-2}:2:-4", "{-2,-4,-6},{-6,-8,-10}"));
}

#endif
