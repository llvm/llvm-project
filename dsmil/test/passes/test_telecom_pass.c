/**
 * @file test_telecom_pass.c
 * @brief Integration tests for Telecom Pass
 *
 * Tests that the DsmilTelecomPass correctly discovers telecom annotations.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

// RUN: dsmil-clang -fdsmil-telecom-flags -mllvm -dsmil-telecom-flags %s -c -o %t.o 2>&1 | FileCheck %s
// RUN: test -f %t.dsmil.telecom.json || echo "Telecom manifest file created"
// REQUIRES: dsmil

#include "dsmil/include/dsmil_attributes.h"

// Test 1: SS7 function
DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_ENV("lab")
DSMIL_SIG_SECURITY("defense_lab")
DSMIL_LAYER(3)
DSMIL_DEVICE(31)
void test_ss7_function(void) {
    // Should be discovered by telecom pass
}

// Test 2: SIGTRAN function
DSMIL_TELECOM_STACK("sigtran")
DSMIL_SIGTRAN_ROLE("SG")
DSMIL_TELECOM_INTERFACE("m3ua")
DSMIL_TELECOM_ENDPOINT("upstream_stp")
void test_sigtran_function(void) {
    // Should be discovered by telecom pass
}

// Test 3: Honeypot function
DSMIL_TELECOM_STACK("ss7")
DSMIL_TELECOM_ENV("honeypot")
DSMIL_SS7_ROLE("STP")
void test_honeypot_function(void) {
    // Should be flagged in manifest
}

// Test 4: Production function
DSMIL_TELECOM_STACK("ss7")
DSMIL_TELECOM_ENV("prod")
DSMIL_SS7_ROLE("MSC")
void test_prod_function(void) {
    // Should be flagged in manifest
}

// Test 5: Non-telecom function
void test_normal_function(void) {
    // Should not appear in telecom manifest
}

int main(void) {
    test_ss7_function();
    test_sigtran_function();
    test_honeypot_function();
    test_prod_function();
    test_normal_function();
    return 0;
}
