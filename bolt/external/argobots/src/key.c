/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/** @defgroup KEY Work-Unit Local Storage (TLS)
 * This group is for work-unit specific data, which can be described as
 * work-unit local storage (TLS).
 */

static ABTD_atomic_uint32 g_key_id =
    ABTD_ATOMIC_UINT32_STATIC_INITIALIZER(ABTI_KEY_ID_END_);

/**
 * @ingroup KEY
 * @brief   Create an WU-specific data key.
 *
 * \c ABT_key_create() creates a new work unit (WU)-specific data key visible
 * to all WUs (ULTs or tasklets) in the process and returns its handle through
 * \c newkey.  Although the same key may be used by different WUs, the values
 * bound to the key by \c ABT_key_set() are maintained per WU and persist for
 * the life of the calling WU.
 *
 * Upon key creation, the value \c NULL shall be associated with the new key in
 * all active WUs.  Upon WU creation, the value \c NULL shall be associated
 * with all defined keys in the new WU.
 *
 * An optional destructor function, \c destructor, may be registered with each
 * key.  When a WU terminates, if a key has a non-NULL destructor pointer, and
 * the WU has a non-NULL value associated with that key, the value of the key
 * is set to \c NULL, and then \c destructor is called with the previously
 * associated value as its sole argument.  The order of destructor calls is
 * unspecified if more than one destructor exists for a WU when it exits.
 *
 * @param[in]  destructor  destructor function called when a WU exits
 * @param[out] newkey      handle to a newly created key
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_key_create(void (*destructor)(void *value), ABT_key *newkey)
{
    ABTI_key *p_newkey;
    int abt_errno = ABTU_malloc(sizeof(ABTI_key), (void **)&p_newkey);
    ABTI_CHECK_ERROR(abt_errno);

    p_newkey->f_destructor = destructor;
    p_newkey->id = ABTD_atomic_fetch_add_uint32(&g_key_id, 1);
    /* Return value */
    *newkey = ABTI_key_get_handle(p_newkey);
    return ABT_SUCCESS;
}

/**
 * @ingroup KEY
 * @brief   Free an WU-specific data key.
 *
 * \c ABT_key_free() deletes the WU-specific data key specified by \c key and
 * deallocates memory used for the key object.  It is the user's responsibility
 * to free memory for values associated with the deleted key.  This routine
 * does not call the destructor function registered by \c ABT_key_create().
 *
 * @param[in,out] key  handle to the target key
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_key_free(ABT_key *key)
{
    ABT_key h_key = *key;
    ABTI_key *p_key = ABTI_key_get_ptr(h_key);
    ABTI_CHECK_NULL_KEY_PTR(p_key);
    ABTU_free(p_key);

    /* Return value */
    *key = ABT_KEY_NULL;
    return ABT_SUCCESS;
}

/**
 * @ingroup KEY
 * @brief   Associate a value with the key.
 *
 * \c ABT_key_set() associates a value, \c value, with the target WU-specific
 * data key, \c key.  Different WUs may bind different values to the same key.
 *
 * @param[in] key    handle to the target key
 * @param[in] value  value for the key
 * @return Error code
 * @retval ABT_SUCCESS          on success
 * @retval ABT_ERR_INV_XSTREAM  called by an external thread
 */
int ABT_key_set(ABT_key key, void *value)
{
    ABTI_key *p_key = ABTI_key_get_ptr(key);
    ABTI_CHECK_NULL_KEY_PTR(p_key);

    ABTI_xstream *p_local_xstream;
    ABTI_SETUP_LOCAL_XSTREAM(&p_local_xstream);

    /* Obtain the key-value table pointer. */
    int abt_errno =
        ABTI_ktable_set(ABTI_xstream_get_local(p_local_xstream),
                        &p_local_xstream->p_thread->p_keytable, p_key, value);
    ABTI_CHECK_ERROR(abt_errno);
    return ABT_SUCCESS;
}

/**
 * @ingroup KEY
 * @brief   Get the value associated with the key.
 *
 * \c ABT_key_get() returns the value associated with the target WU-specific
 * data key, \c key, through \c value on behalf of the calling WU.  Different
 * WUs get different values for the target key via this routine if they have
 * set different values with \c ABT_key_set().  If a WU has never set a value
 * for the key, this routine returns \c NULL to \c value.
 *
 * @param[in] key    handle to the target key
 * @param[in] value  value for the key
 * @return Error code
 * @retval ABT_SUCCESS          on success
 * @retval ABT_ERR_INV_XSTREAM  called by an external thread
 */
int ABT_key_get(ABT_key key, void **value)
{
    ABTI_key *p_key = ABTI_key_get_ptr(key);
    ABTI_CHECK_NULL_KEY_PTR(p_key);

    /* We don't allow an external thread to call this routine. */
    ABTI_xstream *p_local_xstream;
    ABTI_SETUP_LOCAL_XSTREAM(&p_local_xstream);

    /* Obtain the key-value table pointer */
    *value = ABTI_ktable_get(&p_local_xstream->p_thread->p_keytable, p_key);
    return ABT_SUCCESS;
}

void ABTI_ktable_free(ABTI_local *p_local, ABTI_ktable *p_ktable)
{
    ABTI_ktelem *p_elem;
    int i;

    for (i = 0; i < p_ktable->size; i++) {
        p_elem =
            (ABTI_ktelem *)ABTD_atomic_relaxed_load_ptr(&p_ktable->p_elems[i]);
        while (p_elem) {
            /* Call the destructor if it exists and the value is not null. */
            if (p_elem->f_destructor && p_elem->value) {
                p_elem->f_destructor(p_elem->value);
            }
            p_elem =
                (ABTI_ktelem *)ABTD_atomic_relaxed_load_ptr(&p_elem->p_next);
        }
    }
    ABTI_ktable_mem_header *p_header =
        (ABTI_ktable_mem_header *)p_ktable->p_used_mem;
    while (p_header) {
        ABTI_ktable_mem_header *p_next = p_header->p_next;
        if (ABTU_likely(p_header->is_from_mempool)) {
            ABTI_mem_free_desc(p_local, (void *)p_header);
        } else {
            ABTU_free(p_header);
        }
        p_header = p_next;
    }
}
