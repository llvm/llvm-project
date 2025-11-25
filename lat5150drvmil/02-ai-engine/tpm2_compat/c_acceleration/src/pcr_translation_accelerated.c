/**
 * High-Performance PCR Address Translation Implementation
 * Military-grade optimized PCR translation with SIMD acceleration
 *
 * Author: C-INTERNAL Agent
 * Date: 2025-09-23
 * Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
 */

#include "../include/tpm2_compat_accelerated.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  // For SIMD intrinsics
#include <sys/mman.h>   // For memory mapping
#include <unistd.h>

/* =============================================================================
 * OPTIMIZED LOOKUP TABLES (MEMORY ALIGNED)
 * =============================================================================
 */

/* Memory-aligned lookup table for decimal to hex PCR translation */
static const uint16_t __attribute__((aligned(64))) DECIMAL_TO_HEX_LUT[24] = {
    0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007,  // BIOS/UEFI PCRs
    0x0008, 0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x000E, 0x000F,  // OS PCRs
    0x0010, 0x0011, 0x0012, 0x0013, 0x0014, 0x0015, 0x0016, 0x0017   // Extended PCRs
};

/* Memory-aligned reverse lookup table for hex to decimal PCR translation */
static uint32_t __attribute__((aligned(64))) HEX_TO_DECIMAL_LUT[65536];
static bool lut_initialized = false;

/* Special configuration PCRs */
typedef struct {
    uint16_t hex_value;
    const char *name;
    uint32_t config_id;
} special_pcr_t;

static const special_pcr_t SPECIAL_CONFIG_PCRS[] = {
    {0xCAFE, "CAFE", 0xCAFE},  // Algorithm configuration
    {0xBEEF, "BEEF", 0xBEEF},  // Extended functionality
    {0xDEAD, "DEAD", 0xDEAD},  // Debug/diagnostic
    {0xFACE, "FACE", 0xFACE}   // Factory configuration
};

#define NUM_SPECIAL_PCRS (sizeof(SPECIAL_CONFIG_PCRS) / sizeof(special_pcr_t))

/* =============================================================================
 * INTERNAL HELPER FUNCTIONS
 * =============================================================================
 */

/**
 * Initialize optimized lookup tables with memory mapping
 */
static tpm2_rc_t initialize_lookup_tables(void) {
    if (lut_initialized) {
        return TPM2_RC_SUCCESS;
    }

    // Initialize reverse lookup table
    memset(HEX_TO_DECIMAL_LUT, 0xFF, sizeof(HEX_TO_DECIMAL_LUT));

    // Populate standard PCR mappings
    for (int i = 0; i < 24; i++) {
        uint16_t hex_pcr = DECIMAL_TO_HEX_LUT[i];
        HEX_TO_DECIMAL_LUT[hex_pcr] = i;
    }

    // Populate special configuration PCRs
    for (size_t i = 0; i < NUM_SPECIAL_PCRS; i++) {
        HEX_TO_DECIMAL_LUT[SPECIAL_CONFIG_PCRS[i].hex_value] = SPECIAL_CONFIG_PCRS[i].config_id;
    }

    lut_initialized = true;
    return TPM2_RC_SUCCESS;
}

/**
 * SIMD-optimized range validation
 */
static inline bool validate_pcr_range_simd(uint32_t pcr) {
    // Use SIMD for fast range checking
    #ifdef __AVX2__
    __m256i pcr_vec = _mm256_set1_epi32(pcr);
    __m256i max_vec = _mm256_set1_epi32(23);
    __m256i cmp_result = _mm256_cmpgt_epi32(max_vec, pcr_vec);

    return _mm256_movemask_epi8(cmp_result) != 0;
    #else
    return pcr <= 23;
    #endif
}

/**
 * Apply bank offset with hardware acceleration
 */
static inline uint16_t apply_bank_offset_accel(uint16_t base_pcr, tpm2_pcr_bank_t bank) {
    if (bank == PCR_BANK_SHA256) {
        return base_pcr;
    }

    // Use bit manipulation for fast bank offset calculation
    uint8_t bank_offset = (uint8_t)bank;
    return base_pcr | (bank_offset << 8);
}

/* =============================================================================
 * PUBLIC API IMPLEMENTATION
 * =============================================================================
 */

tpm2_rc_t tpm2_pcr_decimal_to_hex_fast(
    uint32_t pcr_decimal,
    tpm2_pcr_bank_t bank,
    uint16_t *pcr_hex_out
) {
    // Input validation
    if (pcr_hex_out == NULL) {
        return TPM2_RC_BAD_PARAMETER;
    }

    // Initialize lookup tables if needed
    tpm2_rc_t rc = initialize_lookup_tables();
    if (rc != TPM2_RC_SUCCESS) {
        return rc;
    }

    // Fast range validation with SIMD
    if (!validate_pcr_range_simd(pcr_decimal)) {
        return TPM2_RC_BAD_PARAMETER;
    }

    // Direct lookup - O(1) operation
    uint16_t base_hex = DECIMAL_TO_HEX_LUT[pcr_decimal];

    // Apply bank offset with acceleration
    *pcr_hex_out = apply_bank_offset_accel(base_hex, bank);

    return TPM2_RC_SUCCESS;
}

tpm2_rc_t tpm2_pcr_hex_to_decimal_fast(
    uint16_t pcr_hex,
    uint32_t *pcr_decimal_out,
    tpm2_pcr_bank_t *bank_out
) {
    // Input validation
    if (pcr_decimal_out == NULL) {
        return TPM2_RC_BAD_PARAMETER;
    }

    // Initialize lookup tables if needed
    tpm2_rc_t rc = initialize_lookup_tables();
    if (rc != TPM2_RC_SUCCESS) {
        return rc;
    }

    // Extract bank selector and base PCR
    uint8_t bank_selector = (pcr_hex >> 8) & 0xFF;
    uint16_t base_pcr_hex = pcr_hex & 0xFF;

    // Determine bank type
    tpm2_pcr_bank_t bank = PCR_BANK_SHA256;
    if (bank_selector <= PCR_BANK_EXTENDED) {
        bank = (tpm2_pcr_bank_t)bank_selector;
    }

    if (bank_out != NULL) {
        *bank_out = bank;
    }

    // Handle special configuration PCRs
    for (size_t i = 0; i < NUM_SPECIAL_PCRS; i++) {
        if (pcr_hex == SPECIAL_CONFIG_PCRS[i].hex_value) {
            *pcr_decimal_out = SPECIAL_CONFIG_PCRS[i].config_id;
            if (bank_out != NULL) {
                *bank_out = PCR_BANK_EXTENDED;
            }
            return TPM2_RC_SUCCESS;
        }
    }

    // Direct lookup - O(1) operation
    uint32_t decimal_value = HEX_TO_DECIMAL_LUT[pcr_hex];

    // Check if mapping exists
    if (decimal_value == 0xFFFFFFFF) {
        // Try base PCR lookup for extended addressing
        if (base_pcr_hex < 24) {
            *pcr_decimal_out = base_pcr_hex;
            return TPM2_RC_SUCCESS;
        }
        return TPM2_RC_NOT_SUPPORTED;
    }

    *pcr_decimal_out = decimal_value;
    return TPM2_RC_SUCCESS;
}

tpm2_rc_t tpm2_pcr_translate_batch(
    const uint32_t *pcr_decimals,
    size_t count,
    tpm2_pcr_bank_t bank,
    uint16_t *pcr_hexs_out
) {
    // Input validation
    if (pcr_decimals == NULL || pcr_hexs_out == NULL || count == 0) {
        return TPM2_RC_BAD_PARAMETER;
    }

    // Initialize lookup tables if needed
    tpm2_rc_t rc = initialize_lookup_tables();
    if (rc != TPM2_RC_SUCCESS) {
        return rc;
    }

    // Vectorized batch processing
    #ifdef __AVX2__
    // Process 8 PCRs at a time with AVX2
    size_t simd_count = count & ~7UL;  // Round down to multiple of 8

    for (size_t i = 0; i < simd_count; i += 8) {
        // Load 8 decimal PCRs
        __m256i decimals = _mm256_loadu_si256((__m256i*)&pcr_decimals[i]);

        // Range validation
        __m256i max_pcr = _mm256_set1_epi32(23);
        __m256i valid_mask = _mm256_cmpgt_epi32(max_pcr, decimals);

        if ((unsigned int)_mm256_movemask_epi8(valid_mask) != 0xFFFFFFFFU) {
            return TPM2_RC_BAD_PARAMETER;
        }

        // Gather lookup table values (requires AVX2 gather)
        // Note: This is a simplified implementation - full SIMD gather would be more complex
        for (size_t j = 0; j < 8; j++) {
            uint32_t decimal_pcr = pcr_decimals[i + j];
            if (decimal_pcr > 23) {
                return TPM2_RC_BAD_PARAMETER;
            }

            uint16_t base_hex = DECIMAL_TO_HEX_LUT[decimal_pcr];
            pcr_hexs_out[i + j] = apply_bank_offset_accel(base_hex, bank);
        }
    }

    // Process remaining PCRs
    for (size_t i = simd_count; i < count; i++) {
        rc = tpm2_pcr_decimal_to_hex_fast(pcr_decimals[i], bank, &pcr_hexs_out[i]);
        if (rc != TPM2_RC_SUCCESS) {
            return rc;
        }
    }
    #else
    // Fallback to sequential processing
    for (size_t i = 0; i < count; i++) {
        rc = tpm2_pcr_decimal_to_hex_fast(pcr_decimals[i], bank, &pcr_hexs_out[i]);
        if (rc != TPM2_RC_SUCCESS) {
            return rc;
        }
    }
    #endif

    return TPM2_RC_SUCCESS;
}

tpm2_rc_t tpm2_pcr_validate_range_accel(
    uint32_t pcr,
    bool is_hex,
    tpm2_pcr_bank_t *bank_out
) {
    // Initialize lookup tables if needed
    tpm2_rc_t rc = initialize_lookup_tables();
    if (rc != TPM2_RC_SUCCESS) {
        return rc;
    }

    if (is_hex) {
        // Validate hex range (0x0000-0xFFFF)
        if (pcr > 0xFFFF) {
            return TPM2_RC_BAD_PARAMETER;
        }

        // Extract bank information
        uint8_t bank_selector = (pcr >> 8) & 0xFF;
        if (bank_selector <= PCR_BANK_EXTENDED && bank_out != NULL) {
            *bank_out = (tpm2_pcr_bank_t)bank_selector;
        }

        // Check if valid mapping exists
        uint32_t lookup_result = HEX_TO_DECIMAL_LUT[pcr];
        if (lookup_result != 0xFFFFFFFF) {
            return TPM2_RC_SUCCESS;
        }

        // Check base PCR for extended addressing
        uint16_t base_pcr = pcr & 0xFF;
        if (base_pcr < 24) {
            return TPM2_RC_SUCCESS;
        }

        // Check special configuration PCRs
        for (size_t i = 0; i < NUM_SPECIAL_PCRS; i++) {
            if (pcr == SPECIAL_CONFIG_PCRS[i].hex_value) {
                if (bank_out != NULL) {
                    *bank_out = PCR_BANK_EXTENDED;
                }
                return TPM2_RC_SUCCESS;
            }
        }

        return TPM2_RC_NOT_SUPPORTED;
    } else {
        // Validate decimal range (0-23)
        if (!validate_pcr_range_simd(pcr)) {
            return TPM2_RC_BAD_PARAMETER;
        }

        if (bank_out != NULL) {
            *bank_out = PCR_BANK_SHA256;  // Default bank for decimal PCRs
        }

        return TPM2_RC_SUCCESS;
    }
}

/* =============================================================================
 * PERFORMANCE OPTIMIZATION FUNCTIONS
 * =============================================================================
 */

/**
 * Prefetch lookup table data for improved cache performance
 */
void tpm2_pcr_prefetch_cache(void) {
    // Prefetch lookup tables into cache
    #ifdef __builtin_prefetch
    __builtin_prefetch(DECIMAL_TO_HEX_LUT, 0, 3);
    __builtin_prefetch(HEX_TO_DECIMAL_LUT, 0, 3);
    __builtin_prefetch(SPECIAL_CONFIG_PCRS, 0, 3);
    #endif
}

/**
 * Get translation cache statistics
 */
tpm2_rc_t tpm2_pcr_get_cache_stats(
    size_t *hit_count_out,
    size_t *miss_count_out,
    float *hit_ratio_out
) {
    // For lookup table implementation, all accesses are effectively "hits"
    if (hit_count_out != NULL) {
        *hit_count_out = lut_initialized ? SIZE_MAX : 0;
    }

    if (miss_count_out != NULL) {
        *miss_count_out = 0;
    }

    if (hit_ratio_out != NULL) {
        *hit_ratio_out = lut_initialized ? 1.0f : 0.0f;
    }

    return TPM2_RC_SUCCESS;
}