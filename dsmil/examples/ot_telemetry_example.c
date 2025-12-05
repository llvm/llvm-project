/**
 * @file ot_telemetry_example.c
 * @brief Example demonstrating OT telemetry instrumentation
 *
 * This example shows how to use DSMIL OT telemetry attributes to
 * instrument OT-critical functions and safety signals.
 *
 * Compile with:
 *   dsmil-clang -fdsmil-ot-telemetry -fdsmil-mission-profile=ics_ops \
 *                -c ot_telemetry_example.c -o ot_telemetry_example.o
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "dsmil/include/dsmil_attributes.h"
#include "dsmil/include/dsmil_ot_telemetry.h"
#include <stdio.h>
#include <stdint.h>

// Safety signal: pressure setpoint for line 7
DSMIL_SAFETY_SIGNAL("line7_pressure_setpoint")
static double pressure_setpoint = 100.0;  // PSI

// Safety signal: flow rate
DSMIL_SAFETY_SIGNAL("line7_flow_rate")
static double flow_rate = 50.0;  // GPM

/**
 * OT-critical function: Pump control update
 * 
 * This function directly controls pump operation, which is critical
 * for OT safety. It's marked as:
 * - OT-critical (requires telemetry)
 * - Authority tier 1 (high-impact control)
 * - SES gate (sends intents to Safety Envelope Supervisor)
 */
DSMIL_OT_CRITICAL
DSMIL_OT_TIER(1)
DSMIL_SES_GATE
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
DSMIL_STAGE("control")
int pump_control_update(int pump_id, double new_pressure) {
    // Update pressure setpoint (automatically logged via safety signal)
    pressure_setpoint = new_pressure;
    
    // Send intent to SES (automatically logged via SES_GATE)
    // In real code, this would call SES API
    printf("Sending SES intent: pump %d pressure = %.2f PSI\n", 
           pump_id, new_pressure);
    
    // Perform control operation
    // (In real implementation, this would interface with pump hardware)
    
    return 0;
}

/**
 * OT-critical function: Valve control
 * 
 * Lower authority tier (optimization/scheduling)
 */
DSMIL_OT_CRITICAL
DSMIL_OT_TIER(2)
DSMIL_LAYER(3)
DSMIL_DEVICE(12)
DSMIL_STAGE("control")
int valve_control_update(int valve_id, double position) {
    // Update valve position
    printf("Valve %d position: %.2f%%\n", valve_id, position);
    
    return 0;
}

/**
 * Analytics function: Monitor system status
 * 
 * Authority tier 3 (analytics/advisory only, no control)
 */
DSMIL_OT_CRITICAL
DSMIL_OT_TIER(3)
DSMIL_LAYER(7)
DSMIL_DEVICE(47)
DSMIL_STAGE("serve")
void monitor_system_status(void) {
    // Read-only monitoring (no control actions)
    printf("System status: pressure=%.2f PSI, flow=%.2f GPM\n",
           pressure_setpoint, flow_rate);
}

/**
 * Main function
 */
int main(int argc, char **argv) {
    // Initialize OT telemetry
    dsmil_ot_telemetry_init();
    
    printf("OT Telemetry Example\n");
    printf("====================\n\n");
    
    // Call OT-critical functions (automatically instrumented)
    pump_control_update(1, 125.5);
    valve_control_update(2, 75.0);
    monitor_system_status();
    
    // Update safety signals (automatically logged)
    pressure_setpoint = 110.0;
    flow_rate = 55.0;
    
    printf("\nTelemetry events should be visible in stderr output\n");
    printf("or via DSMIL_OT_TELEMETRY environment variable control.\n");
    
    dsmil_ot_telemetry_shutdown();
    return 0;
}
