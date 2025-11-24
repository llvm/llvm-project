/**
 * @file border_ops_example.c
 * @brief Example LLM worker for border operations deployment
 *
 * This example demonstrates a minimal LLM inference worker compiled
 * with the border_ops mission profile for maximum security.
 *
 * Mission Profile: border_ops
 * Classification: RESTRICTED
 * Deployment: Air-gapped border stations
 *
 * Compile:
 *   dsmil-clang -fdsmil-mission-profile=border_ops \
 *     -fdsmil-provenance=full -O3 border_ops_example.c \
 *     -o border_ops_worker
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <dsmil_attributes.h>
#include <stdint.h>
#include <stdio.h>

// Forward declarations
int llm_inference_loop(void);
void process_query(const uint8_t *input, size_t len, uint8_t *output);
void derive_session_key(const uint8_t *master, uint8_t *session);

/**
 * Main entry point - border operations profile
 * This function is annotated with border_ops mission profile and
 * uses the combined LLM_WORKER_MAIN macro for typical settings.
 */
DSMIL_MISSION_PROFILE("border_ops")
DSMIL_LLM_WORKER_MAIN  // Layer 7, Device 47, serve stage, strict sandbox
int main(int argc, char **argv) {
    printf("[Border Ops Worker] Starting LLM inference service\n");
    printf("[Border Ops Worker] Mission Profile: border_ops\n");
    printf("[Border Ops Worker] Classification: RESTRICTED\n");
    printf("[Border Ops Worker] Mode: Air-gapped, local inference only\n");

    return llm_inference_loop();
}

/**
 * Main inference loop
 * Runs on NPU (Device 47) in Layer 7 (AI/ML Applications)
 */
DSMIL_STAGE("serve")
DSMIL_LAYER(7)
DSMIL_DEVICE(47)  // NPU primary (whitelisted in border_ops)
DSMIL_ROE("ANALYSIS_ONLY")
int llm_inference_loop(void) {
    // Simulated inference loop
    uint8_t input[1024];
    uint8_t output[1024];

    for (int i = 0; i < 10; i++) {
        // In real implementation, would read from secure IPC channel
        process_query(input, sizeof(input), output);
    }

    printf("[Border Ops Worker] Inference loop completed\n");
    return 0;
}

/**
 * Process LLM query
 * Marked as production "serve" stage - debug stages not allowed in border_ops
 */
DSMIL_STAGE("serve")
DSMIL_LAYER(7)
DSMIL_DEVICE(47)
void process_query(const uint8_t *input, size_t len, uint8_t *output) {
    // Quantized INT8 inference on NPU
    // In real implementation, would call NPU kernels

    // Simulate processing
    for (size_t i = 0; i < len && i < 16; i++) {
        output[i] = input[i] ^ 0xAA;
    }
}

/**
 * Derive session key using constant-time crypto
 * This function is marked as DSMIL_SECRET to enforce constant-time execution
 * to prevent timing side-channel attacks.
 *
 * Runs on Layer 3 (Crypto Services) using dedicated crypto engine (Device 30)
 */
DSMIL_SECRET
DSMIL_LAYER(3)
DSMIL_DEVICE(30)  // Crypto engine (whitelisted in border_ops)
DSMIL_ROE("CRYPTO_SIGN")
void derive_session_key(const uint8_t *master, uint8_t *session) {
    // Constant-time key derivation (HKDF or similar)
    // The DSMIL_SECRET attribute ensures:
    // - No secret-dependent branches
    // - No secret-dependent memory access
    // - No variable-time instructions on secrets

    // Simplified constant-time XOR (real implementation would use HKDF)
    for (int i = 0; i < 32; i++) {
        session[i] = master[i] ^ 0x5C;  // Constant-time operation
    }
}

/**
 * Example of INVALID code for border_ops profile
 *
 * The following functions would cause compile-time errors:
 */

#if 0  // Disabled - these would fail to compile

// ERROR: Stage "debug" not allowed in border_ops
DSMIL_MISSION_PROFILE("border_ops")
DSMIL_STAGE("debug")  // Compile error!
void debug_print_state(void) {
    // Debug code not allowed in border_ops
}

// ERROR: Device 40 (GPU) not whitelisted in border_ops
DSMIL_MISSION_PROFILE("border_ops")
DSMIL_DEVICE(40)  // Compile error! GPU not whitelisted
void gpu_inference(void) {
    // GPU not allowed in border_ops
}

// ERROR: Quantum export forbidden in border_ops
DSMIL_MISSION_PROFILE("border_ops")
DSMIL_QUANTUM_CANDIDATE("placement")  // Compile error!
int quantum_optimize(void) {
    // Quantum features not allowed in border_ops
}

#endif  // End of invalid examples

/**
 * Compilation and Verification:
 *
 * $ dsmil-clang -fdsmil-mission-profile=border_ops \
 *     -fdsmil-provenance=full -fdsmil-provenance-sign-key=tpm://dsmil \
 *     -O3 border_ops_example.c -o border_ops_worker
 *
 * [DSMIL Mission Policy] Enforcing mission profile: border_ops (Border Operations)
 *   Classification: RESTRICTED
 *   CT Enforcement: strict
 *   Telemetry Level: minimal
 * [DSMIL CT Check] Verifying constant-time enforcement...
 * [DSMIL CT Check] ✓ Function 'derive_session_key' is constant-time
 * [DSMIL Mission Policy] ✓ All functions comply with mission profile
 * [DSMIL Provenance] Signing with ML-DSA-87 (TPM key)
 *
 * $ dsmil-inspect border_ops_worker
 * Mission Profile: border_ops
 * Classification: RESTRICTED
 * Compiled: 2026-01-15T14:30:00Z
 * Signature: VALID (ML-DSA-87, TPM key)
 * Devices: [0, 1, 2, 3, 30, 31, 32, 33, 47, 50, 53]
 * Expiration: None
 * Status: DEPLOYABLE
 */
