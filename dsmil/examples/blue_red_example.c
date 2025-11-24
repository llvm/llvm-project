/**
 * @file blue_red_example.c
 * @brief DSLLVM Blue vs Red Scenario Simulation Example (Feature 2.3)
 *
 * Demonstrates dual-build instrumentation for adversarial testing.
 *
 * Blue build (production):
 *   dsmil-clang -fdsmil-role=blue -O3 -o blue.bin blue_red_example.c
 *
 * Red build (testing):
 *   dsmil-clang -fdsmil-role=red -O3 -o red.bin blue_red_example.c
 *   DSMIL_RED_SCENARIOS="bypass_validation" ./red.bin
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <dsmil_attributes.h>
#include <stdio.h>
#include <string.h>

// Example 1: Red team hook for injection point
DSMIL_RED_TEAM_HOOK("user_input_injection")
DSMIL_ATTACK_SURFACE
void process_user_input(const char *input) {
    #ifdef DSMIL_RED_BUILD
        extern void dsmil_red_log(const char*, const char*);
        extern int dsmil_red_scenario(const char*);

        dsmil_red_log("user_input_processing", __func__);

        // Red build: simulate bypassing validation
        if (dsmil_red_scenario("bypass_validation")) {
            printf("[RED] Simulating validation bypass\n");
            printf("[RED] Processing untrusted input: %s\n", input);
            return;  // Skip validation
        }
    #endif

    // Normal path: validate input
    if (strlen(input) > 100) {
        printf("[BLUE] Input too long, rejecting\n");
        return;
    }
    printf("[BLUE] Processing validated input\n");
}

// Example 2: Vulnerability injection point
DSMIL_VULN_INJECT("buffer_overflow")
void copy_data(char *dest, const char *src, size_t len) {
    #ifdef DSMIL_RED_BUILD
        extern int dsmil_red_scenario(const char*);

        if (dsmil_red_scenario("trigger_overflow")) {
            printf("[RED] Simulating buffer overflow\n");
            memcpy(dest, src, len + 100);  // Intentional overflow
            return;
        }
    #endif

    // Normal path: safe copy
    memcpy(dest, src, len);
}

// Example 3: Blast radius tracking
DSMIL_BLAST_RADIUS
DSMIL_LAYER(8)
void critical_security_operation(void) {
    printf("Executing critical security operation\n");
    // If compromised in red build, analyze blast radius
}

// Main entry point
DSMIL_BUILD_ROLE("blue")
int main(int argc, char **argv) {
    #ifdef DSMIL_RED_BUILD
        extern int dsmil_blue_red_init(int);
        extern void dsmil_blue_red_shutdown(void);

        printf("\n=== RED TEAM BUILD ===\n");
        printf("FOR TESTING ONLY - NEVER DEPLOY\n\n");

        dsmil_blue_red_init(1);
    #else
        printf("=== BLUE TEAM BUILD ===\n");
        printf("Production configuration\n\n");
    #endif

    // Test scenarios
    process_user_input("test input");

    char dest[64];
    copy_data(dest, "source data", 11);

    critical_security_operation();

    #ifdef DSMIL_RED_BUILD
        dsmil_blue_red_shutdown();
    #endif

    return 0;
}
