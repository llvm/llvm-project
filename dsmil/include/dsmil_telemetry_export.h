/**
 * @file dsmil_telemetry_export.h
 * @brief Runtime Telemetry Export API
 *
 * Provides standardized telemetry export for Prometheus, OpenTelemetry,
 * and structured logging (ELK/Splunk).
 *
 * Version: 1.7.0
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef DSMIL_TELEMETRY_EXPORT_H
#define DSMIL_TELEMETRY_EXPORT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup DSMIL_TELEMETRY_EXPORT Telemetry Export
 * @{
 */

/**
 * @brief Telemetry export format
 */
typedef enum {
    DSMIL_TELEMETRY_PROMETHEUS,
    DSMIL_TELEMETRY_OPENTELEMETRY,
    DSMIL_TELEMETRY_JSON,
    DSMIL_TELEMETRY_NONE
} dsmil_telemetry_format_t;

/**
 * @brief Telemetry export options
 */
typedef struct {
    dsmil_telemetry_format_t format;
    const char *endpoint;
    int port;
    const char *output_file;
    bool enable_performance;
    bool enable_security;
    bool enable_operational;
} dsmil_telemetry_options_t;

/**
 * @brief Initialize telemetry export system
 *
 * @param options Telemetry options
 * @return 0 on success, -1 on error
 */
int dsmil_telemetry_init(const dsmil_telemetry_options_t *options);

/**
 * @brief Record performance metric
 *
 * @param name Metric name
 * @param value Metric value
 * @param labels Optional labels (NULL-terminated array)
 */
void dsmil_telemetry_record_counter(const char *name, uint64_t value, const char **labels);

/**
 * @brief Record gauge metric
 *
 * @param name Metric name
 * @param value Current value
 * @param labels Optional labels
 */
void dsmil_telemetry_record_gauge(const char *name, double value, const char **labels);

/**
 * @brief Record histogram metric
 *
 * @param name Metric name
 * @param value Value to record
 * @param labels Optional labels
 */
void dsmil_telemetry_record_histogram(const char *name, double value, const char **labels);

/**
 * @brief Record security event
 *
 * @param event_type Event type (e.g., "classification_cross", "provenance_verify")
 * @param severity Severity level (0-10)
 * @param details Event details JSON string
 */
void dsmil_telemetry_record_security_event(const char *event_type,
                                            int severity,
                                            const char *details);

/**
 * @brief Record operational metric
 *
 * @param name Metric name
 * @param value Metric value
 * @param labels Optional labels
 */
void dsmil_telemetry_record_operational(const char *name, uint64_t value, const char **labels);

/**
 * @brief Flush telemetry data
 *
 * Forces immediate export of buffered telemetry data.
 *
 * @return 0 on success, -1 on error
 */
int dsmil_telemetry_flush(void);

/**
 * @brief Shutdown telemetry export system
 *
 * @return 0 on success, -1 on error
 */
int dsmil_telemetry_shutdown(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* DSMIL_TELEMETRY_EXPORT_H */
