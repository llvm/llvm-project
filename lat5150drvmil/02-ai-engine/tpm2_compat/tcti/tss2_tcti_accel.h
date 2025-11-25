/**
 * TSS2 TCTI Acceleration Plugin - Header File
 *
 * Provides transparent hardware-accelerated TPM operations via TSS2 TCTI interface
 */

#ifndef TSS2_TCTI_ACCEL_H
#define TSS2_TCTI_ACCEL_H

#include <tss2/tss2_tcti.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize TCTI Acceleration Context
 *
 * @param context TCTI context to initialize (or NULL to query size)
 * @param size Pointer to size variable
 * @param config Configuration string (format: "device=PATH,accel=FLAGS,security=LEVEL")
 * @return TSS2_RC_SUCCESS on success, error code otherwise
 */
TSS2_RC Tss2_Tcti_Accel_Init(
    TSS2_TCTI_CONTEXT *context,
    size_t *size,
    const char *config
);

/**
 * Get TCTI info structure
 *
 * @return Pointer to TCTI info structure
 */
const TSS2_TCTI_INFO* Tss2_Tcti_Info(void);

#ifdef __cplusplus
}
#endif

#endif /* TSS2_TCTI_ACCEL_H */
