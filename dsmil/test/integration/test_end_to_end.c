/**
 * @file test_end_to_end.c
 * @brief End-to-end integration tests
 *
 * Tests complete workflows combining multiple DSLLVM features.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_ot_telemetry.h"
#include "dsmil/include/dsmil_fuzz_telemetry.h"
#include "dsmil/include/dsmil_telecom_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test 1: OT telemetry workflow
DSMIL_OT_CRITICAL
DSMIL_OT_TIER(1)
DSMIL_SES_GATE
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
DSMIL_STAGE("control")
DSMIL_SAFETY_SIGNAL("pressure")
static double pressure = 100.0;

void test_ot_workflow(void) {
    dsmil_ot_telemetry_init();
    
    // Simulate OT-critical operation
    pressure = 125.0;
    
    dsmil_telemetry_event_t ev = {
        .event_type = DSMIL_TELEMETRY_OT_PATH_ENTRY,
        .module_id = "pump_controller",
        .func_id = "update_pressure",
        .file = __FILE__,
        .line = __LINE__,
        .layer = 3,
        .device = 12,
        .stage = "control",
        .mission_profile = "ics_ops",
        .authority_tier = 1,
        .signal_name = "pressure",
        .signal_value = pressure,
        .signal_min = 50.0,
        .signal_max = 200.0
    };
    
    dsmil_telemetry_event(&ev);
    
    dsmil_ot_telemetry_shutdown();
}

// Test 2: Telecom workflow
DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
DSMIL_TELECOM_ENV("lab")
void test_telecom_workflow(void) {
    dsmil_ot_telemetry_init();
    
    // Simulate SS7 message processing
    DSMIL_LOG_SS7_RX(0x1234, 0x5678, 0x08, 1, 2);
    
    // Process message...
    
    DSMIL_LOG_SS7_TX(0x1234, 0x5678, 0x08, 1, 2);
    
    dsmil_ot_telemetry_shutdown();
}

// Test 3: Fuzzing workflow
DSMIL_FUZZ_COVERAGE
DSMIL_FUZZ_ENTRY_POINT
DSMIL_FUZZ_STATE_MACHINE("parser")
int test_fuzzing_workflow(const uint8_t *data, size_t len) {
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    dsmil_fuzz_set_context(0x12345678);
    
    int state = 0;
    for (size_t i = 0; i < len; i++) {
        dsmil_fuzz_cov_hit(i);
        
        if (data[i] == 'A') {
            state = 1;
            dsmil_fuzz_state_transition(1, 0, 1);
        } else if (data[i] == 'B') {
            state = 2;
            dsmil_fuzz_state_transition(1, 1, 2);
        }
    }
    
    dsmil_fuzz_metric_record("parse", 10, 20, 5, 1000);
    
    dsmil_fuzz_telemetry_shutdown();
    
    return state;
}

// Test 4: Combined OT + Telecom workflow
DSMIL_OT_CRITICAL
DSMIL_TELECOM_STACK("ss7")
DSMIL_SS7_ROLE("STP")
void test_combined_ot_telecom(void) {
    dsmil_ot_telemetry_init();
    
    // OT-critical telecom operation
    dsmil_telemetry_event_t ev = {
        .event_type = DSMIL_TELEMETRY_SS7_MSG_RX,
        .module_id = "ss7_handler",
        .func_id = "process_message",
        .file = __FILE__,
        .line = __LINE__,
        .layer = 3,
        .device = 31,
        .telecom_stack = "ss7",
        .ss7_role = "STP",
        .telecom_env = "lab",
        .ss7_opc = 0x1234,
        .ss7_dpc = 0x5678,
        .ss7_sio = 0x08
    };
    
    dsmil_telemetry_event(&ev);
    
    dsmil_ot_telemetry_shutdown();
}

// Test 5: Fuzzing with OT awareness
DSMIL_FUZZ_COVERAGE
DSMIL_OT_CRITICAL
void test_fuzzing_with_ot(const uint8_t *data, size_t len) {
    dsmil_ot_telemetry_init();
    dsmil_fuzz_telemetry_init(NULL, 1024);
    
    // Fuzzing operation with OT awareness
    for (size_t i = 0; i < len; i++) {
        dsmil_fuzz_cov_hit(i);
    }
    
    dsmil_telemetry_event_t ev = {
        .event_type = DSMIL_TELEMETRY_OT_PATH_ENTRY,
        .module_id = "fuzz_target",
        .func_id = "test_fuzzing_with_ot",
        .file = __FILE__,
        .line = __LINE__,
        .layer = 3,
        .device = 12
    };
    dsmil_telemetry_event(&ev);
    
    dsmil_fuzz_telemetry_shutdown();
    dsmil_ot_telemetry_shutdown();
}

int main(void) {
    printf("Running end-to-end integration tests...\n");
    
    test_ot_workflow();
    printf("✓ OT workflow test passed\n");
    
    test_telecom_workflow();
    printf("✓ Telecom workflow test passed\n");
    
    uint8_t test_data[] = "AB";
    test_fuzzing_workflow(test_data, sizeof(test_data));
    printf("✓ Fuzzing workflow test passed\n");
    
    test_combined_ot_telecom();
    printf("✓ Combined OT+Telecom test passed\n");
    
    test_fuzzing_with_ot(test_data, sizeof(test_data));
    printf("✓ Fuzzing with OT test passed\n");
    
    printf("\nAll end-to-end tests passed!\n");
    return 0;
}
