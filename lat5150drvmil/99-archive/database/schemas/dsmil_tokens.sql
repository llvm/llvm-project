-- DSMIL Token Testing Database Schema
-- Comprehensive data recording system for Dell Latitude 5450 MIL-SPEC
-- Support for 72 DSMIL tokens across 6 groups of 12 devices each
-- Version: 1.0.0
-- Date: 2025-09-01

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = MEMORY;

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Test Sessions: Top-level test execution tracking
CREATE TABLE test_sessions (
    session_id TEXT PRIMARY KEY,
    session_name TEXT NOT NULL,
    start_timestamp REAL NOT NULL,
    end_timestamp REAL,
    session_type TEXT NOT NULL, -- single, group, range, comprehensive
    total_tokens INTEGER DEFAULT 0,
    successful_tokens INTEGER DEFAULT 0,
    failed_tokens INTEGER DEFAULT 0,
    emergency_stops INTEGER DEFAULT 0,
    thermal_warnings INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running', -- running, completed, failed, aborted
    operator TEXT,
    notes TEXT,
    created_at REAL NOT NULL DEFAULT (julianday('now')),
    updated_at REAL NOT NULL DEFAULT (julianday('now'))
);

-- Token Definitions: Master token reference data
CREATE TABLE token_definitions (
    token_id INTEGER PRIMARY KEY,
    hex_id TEXT UNIQUE NOT NULL,
    group_id INTEGER NOT NULL CHECK (group_id >= 0 AND group_id < 6),
    device_id INTEGER NOT NULL CHECK (device_id >= 0 AND device_id < 12),
    sequential_index INTEGER NOT NULL CHECK (sequential_index >= 0 AND sequential_index < 72),
    potential_function TEXT,
    confidence REAL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    description TEXT,
    accessible BOOLEAN DEFAULT NULL,
    security_level TEXT DEFAULT 'unknown', -- low, medium, high, critical, unknown
    thermal_impact TEXT DEFAULT 'unknown', -- none, low, medium, high, critical, unknown
    created_at REAL NOT NULL DEFAULT (julianday('now')),
    updated_at REAL NOT NULL DEFAULT (julianday('now')),
    UNIQUE (group_id, device_id)
);

-- Token Tests: Individual token test operations
CREATE TABLE token_tests (
    test_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    token_id INTEGER NOT NULL,
    test_timestamp REAL NOT NULL,
    access_method TEXT NOT NULL, -- smbios-token-ctl, smi, direct, wmi
    operation_type TEXT NOT NULL, -- read, write, toggle, probe, activate
    initial_value TEXT,
    set_value TEXT,
    final_value TEXT,
    expected_value TEXT,
    test_duration_ms INTEGER,
    success BOOLEAN NOT NULL,
    error_code TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    notes TEXT,
    created_at REAL NOT NULL DEFAULT (julianday('now')),
    FOREIGN KEY (session_id) REFERENCES test_sessions (session_id),
    FOREIGN KEY (token_id) REFERENCES token_definitions (token_id)
);

-- System Metrics: System performance during testing
CREATE TABLE system_metrics (
    metric_id TEXT PRIMARY KEY,
    test_id TEXT,
    session_id TEXT NOT NULL,
    metric_timestamp REAL NOT NULL,
    cpu_percent REAL,
    memory_percent REAL,
    memory_available_gb REAL,
    disk_usage_percent REAL,
    system_load_1min REAL,
    system_load_5min REAL,
    system_load_15min REAL,
    uptime_hours REAL,
    process_count INTEGER,
    created_at REAL NOT NULL DEFAULT (julianday('now')),
    FOREIGN KEY (test_id) REFERENCES token_tests (test_id),
    FOREIGN KEY (session_id) REFERENCES test_sessions (session_id)
);

-- Thermal Readings: Temperature monitoring data
CREATE TABLE thermal_readings (
    reading_id TEXT PRIMARY KEY,
    test_id TEXT,
    session_id TEXT NOT NULL,
    reading_timestamp REAL NOT NULL,
    sensor_name TEXT NOT NULL,
    temperature_celsius REAL NOT NULL,
    critical_temp REAL,
    warning_temp REAL,
    thermal_state TEXT, -- normal, warning, critical, emergency
    fan_speed_rpm INTEGER,
    thermal_throttling BOOLEAN DEFAULT FALSE,
    created_at REAL NOT NULL DEFAULT (julianday('now')),
    FOREIGN KEY (test_id) REFERENCES token_tests (test_id),
    FOREIGN KEY (session_id) REFERENCES test_sessions (session_id)
);

-- Kernel Messages: dmesg and kernel log capture
CREATE TABLE kernel_messages (
    message_id TEXT PRIMARY KEY,
    test_id TEXT,
    session_id TEXT NOT NULL,
    message_timestamp REAL NOT NULL,
    log_level TEXT NOT NULL, -- emerg, alert, crit, err, warn, notice, info, debug
    subsystem TEXT, -- dsmil, thermal, smbios, acpi, etc
    message_text TEXT NOT NULL,
    raw_message TEXT NOT NULL,
    correlation_tags TEXT, -- JSON array of correlation tags
    created_at REAL NOT NULL DEFAULT (julianday('now')),
    FOREIGN KEY (test_id) REFERENCES token_tests (test_id),
    FOREIGN KEY (session_id) REFERENCES test_sessions (session_id)
);

-- DSMIL Device Responses: Kernel module device state tracking
CREATE TABLE dsmil_responses (
    response_id TEXT PRIMARY KEY,
    test_id TEXT,
    session_id TEXT NOT NULL,
    response_timestamp REAL NOT NULL,
    group_id INTEGER NOT NULL,
    device_id INTEGER,
    response_type TEXT NOT NULL, -- activation, deactivation, state_change, error, memory_map
    previous_state TEXT,
    new_state TEXT,
    response_data TEXT, -- JSON or text data
    memory_address TEXT,
    memory_size INTEGER,
    correlation_strength REAL, -- 0.0 to 1.0 correlation to token operation
    response_delay_ms INTEGER, -- delay from token operation to response
    created_at REAL NOT NULL DEFAULT (julianday('now')),
    FOREIGN KEY (test_id) REFERENCES token_tests (test_id),
    FOREIGN KEY (session_id) REFERENCES test_sessions (session_id)
);

-- Token Correlations: Relationship tracking between tokens and responses
CREATE TABLE token_correlations (
    correlation_id TEXT PRIMARY KEY,
    primary_token_id INTEGER NOT NULL,
    correlated_token_id INTEGER,
    correlation_type TEXT NOT NULL, -- sequential, simultaneous, dependent, conflicting
    correlation_strength REAL NOT NULL CHECK (correlation_strength >= 0.0 AND correlation_strength <= 1.0),
    correlation_window_ms INTEGER, -- time window for correlation
    test_count INTEGER DEFAULT 1,
    success_rate REAL, -- percentage of successful correlations
    notes TEXT,
    discovered_at REAL NOT NULL DEFAULT (julianday('now')),
    last_observed REAL NOT NULL DEFAULT (julianday('now')),
    FOREIGN KEY (primary_token_id) REFERENCES token_definitions (token_id),
    FOREIGN KEY (correlated_token_id) REFERENCES token_definitions (token_id)
);

-- Discovery Log: Track discovered functionality and behaviors
CREATE TABLE discovery_log (
    discovery_id TEXT PRIMARY KEY,
    test_id TEXT,
    session_id TEXT,
    discovery_timestamp REAL NOT NULL,
    discovery_type TEXT NOT NULL, -- functionality, behavior, correlation, error_pattern, security
    token_ids TEXT, -- JSON array of related token IDs
    discovery_title TEXT NOT NULL,
    discovery_description TEXT NOT NULL,
    evidence_data TEXT, -- JSON data supporting the discovery
    confidence_level REAL CHECK (confidence_level >= 0.0 AND confidence_level <= 1.0),
    verified BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at REAL NOT NULL DEFAULT (julianday('now')),
    FOREIGN KEY (test_id) REFERENCES token_tests (test_id),
    FOREIGN KEY (session_id) REFERENCES test_sessions (session_id)
);

-- ============================================================================
-- INDICES FOR PERFORMANCE
-- ============================================================================

-- Session indices
CREATE INDEX idx_sessions_timestamp ON test_sessions (start_timestamp);
CREATE INDEX idx_sessions_type ON test_sessions (session_type);
CREATE INDEX idx_sessions_status ON test_sessions (status);

-- Token definition indices
CREATE INDEX idx_tokens_group ON token_definitions (group_id);
CREATE INDEX idx_tokens_accessible ON token_definitions (accessible);
CREATE INDEX idx_tokens_function ON token_definitions (potential_function);

-- Token test indices
CREATE INDEX idx_tests_session ON token_tests (session_id);
CREATE INDEX idx_tests_token ON token_tests (token_id);
CREATE INDEX idx_tests_timestamp ON token_tests (test_timestamp);
CREATE INDEX idx_tests_success ON token_tests (success);
CREATE INDEX idx_tests_session_token ON token_tests (session_id, token_id);

-- System metrics indices
CREATE INDEX idx_metrics_session ON system_metrics (session_id);
CREATE INDEX idx_metrics_timestamp ON system_metrics (metric_timestamp);
CREATE INDEX idx_metrics_test ON system_metrics (test_id);

-- Thermal reading indices
CREATE INDEX idx_thermal_session ON thermal_readings (session_id);
CREATE INDEX idx_thermal_timestamp ON thermal_readings (reading_timestamp);
CREATE INDEX idx_thermal_sensor ON thermal_readings (sensor_name);
CREATE INDEX idx_thermal_state ON thermal_readings (thermal_state);

-- Kernel message indices
CREATE INDEX idx_kernel_session ON kernel_messages (session_id);
CREATE INDEX idx_kernel_timestamp ON kernel_messages (message_timestamp);
CREATE INDEX idx_kernel_level ON kernel_messages (log_level);
CREATE INDEX idx_kernel_subsystem ON kernel_messages (subsystem);

-- DSMIL response indices
CREATE INDEX idx_dsmil_session ON dsmil_responses (session_id);
CREATE INDEX idx_dsmil_timestamp ON dsmil_responses (response_timestamp);
CREATE INDEX idx_dsmil_group ON dsmil_responses (group_id);
CREATE INDEX idx_dsmil_type ON dsmil_responses (response_type);
CREATE INDEX idx_dsmil_correlation ON dsmil_responses (correlation_strength);

-- Correlation indices
CREATE INDEX idx_corr_primary ON token_correlations (primary_token_id);
CREATE INDEX idx_corr_secondary ON token_correlations (correlated_token_id);
CREATE INDEX idx_corr_strength ON token_correlations (correlation_strength);
CREATE INDEX idx_corr_type ON token_correlations (correlation_type);

-- Discovery log indices
CREATE INDEX idx_discovery_session ON discovery_log (session_id);
CREATE INDEX idx_discovery_timestamp ON discovery_log (discovery_timestamp);
CREATE INDEX idx_discovery_type ON discovery_log (discovery_type);
CREATE INDEX idx_discovery_verified ON discovery_log (verified);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Session summary view
CREATE VIEW session_summary AS
SELECT 
    s.session_id,
    s.session_name,
    s.session_type,
    s.status,
    datetime(s.start_timestamp) as start_time,
    datetime(s.end_timestamp) as end_time,
    s.total_tokens,
    s.successful_tokens,
    s.failed_tokens,
    ROUND((s.successful_tokens * 100.0 / NULLIF(s.total_tokens, 0)), 2) as success_rate,
    s.emergency_stops,
    s.thermal_warnings,
    COUNT(DISTINCT tt.token_id) as unique_tokens_tested,
    MAX(tr.temperature_celsius) as max_temperature,
    AVG(sm.cpu_percent) as avg_cpu_usage,
    AVG(sm.memory_percent) as avg_memory_usage
FROM test_sessions s
LEFT JOIN token_tests tt ON s.session_id = tt.session_id
LEFT JOIN thermal_readings tr ON s.session_id = tr.session_id
LEFT JOIN system_metrics sm ON s.session_id = sm.session_id
GROUP BY s.session_id;

-- Token success rate view
CREATE VIEW token_success_rates AS
SELECT 
    td.token_id,
    td.hex_id,
    td.group_id,
    td.device_id,
    td.potential_function,
    COUNT(tt.test_id) as total_tests,
    SUM(CASE WHEN tt.success THEN 1 ELSE 0 END) as successful_tests,
    ROUND((SUM(CASE WHEN tt.success THEN 1 ELSE 0 END) * 100.0 / COUNT(tt.test_id)), 2) as success_rate,
    AVG(tt.test_duration_ms) as avg_duration_ms,
    MAX(tt.test_timestamp) as last_tested
FROM token_definitions td
LEFT JOIN token_tests tt ON td.token_id = tt.token_id
GROUP BY td.token_id;

-- Group performance view
CREATE VIEW group_performance AS
SELECT 
    td.group_id,
    COUNT(DISTINCT td.token_id) as tokens_in_group,
    COUNT(DISTINCT tt.test_id) as total_tests,
    SUM(CASE WHEN tt.success THEN 1 ELSE 0 END) as successful_tests,
    ROUND((SUM(CASE WHEN tt.success THEN 1 ELSE 0 END) * 100.0 / COUNT(tt.test_id)), 2) as group_success_rate,
    COUNT(DISTINCT dr.response_id) as dsmil_responses,
    AVG(dr.correlation_strength) as avg_correlation_strength,
    MAX(tr.temperature_celsius) as max_group_temperature
FROM token_definitions td
LEFT JOIN token_tests tt ON td.token_id = tt.token_id
LEFT JOIN dsmil_responses dr ON tt.test_id = dr.test_id
LEFT JOIN thermal_readings tr ON tt.test_id = tr.test_id
GROUP BY td.group_id;

-- Thermal impact analysis view
CREATE VIEW thermal_impact_analysis AS
SELECT 
    td.token_id,
    td.hex_id,
    td.group_id,
    AVG(tr_before.temperature_celsius) as avg_temp_before,
    AVG(tr_after.temperature_celsius) as avg_temp_after,
    AVG(tr_after.temperature_celsius) - AVG(tr_before.temperature_celsius) as temp_delta,
    MAX(tr_after.temperature_celsius) as peak_temperature,
    COUNT(CASE WHEN tr_after.thermal_state IN ('warning', 'critical', 'emergency') THEN 1 END) as thermal_events
FROM token_definitions td
JOIN token_tests tt ON td.token_id = tt.token_id
LEFT JOIN thermal_readings tr_before ON tt.test_id = tr_before.test_id 
    AND tr_before.reading_timestamp < tt.test_timestamp
LEFT JOIN thermal_readings tr_after ON tt.test_id = tr_after.test_id 
    AND tr_after.reading_timestamp > tt.test_timestamp
    AND tr_after.reading_timestamp < tt.test_timestamp + (30.0 / 86400.0) -- 30 seconds after
GROUP BY td.token_id
HAVING COUNT(tr_before.reading_id) > 0 AND COUNT(tr_after.reading_id) > 0;

-- DSMIL correlation strength view
CREATE VIEW dsmil_correlation_analysis AS
SELECT 
    td.token_id,
    td.hex_id,
    td.group_id,
    td.device_id,
    COUNT(dr.response_id) as total_responses,
    AVG(dr.correlation_strength) as avg_correlation_strength,
    MAX(dr.correlation_strength) as max_correlation_strength,
    AVG(dr.response_delay_ms) as avg_response_delay,
    COUNT(DISTINCT dr.response_type) as unique_response_types,
    GROUP_CONCAT(DISTINCT dr.response_type) as response_types
FROM token_definitions td
JOIN token_tests tt ON td.token_id = tt.token_id
JOIN dsmil_responses dr ON tt.test_id = dr.test_id
GROUP BY td.token_id;

-- ============================================================================
-- TRIGGERS FOR AUTOMATION
-- ============================================================================

-- Auto-update session statistics
CREATE TRIGGER update_session_stats
AFTER INSERT ON token_tests
FOR EACH ROW
BEGIN
    UPDATE test_sessions 
    SET total_tokens = total_tokens + 1,
        successful_tokens = successful_tokens + CASE WHEN NEW.success THEN 1 ELSE 0 END,
        failed_tokens = failed_tokens + CASE WHEN NEW.success THEN 0 ELSE 1 END,
        updated_at = julianday('now')
    WHERE session_id = NEW.session_id;
END;

-- Update timestamps on record updates
CREATE TRIGGER update_session_timestamp
AFTER UPDATE ON test_sessions
FOR EACH ROW
BEGIN
    UPDATE test_sessions 
    SET updated_at = julianday('now')
    WHERE session_id = NEW.session_id;
END;

CREATE TRIGGER update_token_timestamp
AFTER UPDATE ON token_definitions
FOR EACH ROW
BEGIN
    UPDATE token_definitions 
    SET updated_at = julianday('now')
    WHERE token_id = NEW.token_id;
END;

-- ============================================================================
-- SAMPLE DATA INSERTION
-- ============================================================================

-- Insert the 72 DSMIL token definitions
INSERT INTO token_definitions (token_id, hex_id, group_id, device_id, sequential_index, potential_function, confidence, description, accessible) VALUES
-- Group 0 (Tokens 1152-1163, 0x480-0x48B)
(1152, '0x480', 0, 0, 0, 'power_management', 0.9, 'Group 0, Device 0: power_management', 0),
(1153, '0x481', 0, 1, 1, 'thermal_control', 0.9, 'Group 0, Device 1: thermal_control', 1),
(1154, '0x482', 0, 2, 2, 'security_module', 0.8, 'Group 0, Device 2: security_module', 1),
(1155, '0x483', 0, 3, 3, 'diagnostic_mode', 0.7, 'Group 0, Device 3: diagnostic_mode', 1),
(1156, '0x484', 0, 4, 4, 'network_interface', 0.8, 'Group 0, Device 4: network_interface', 1),
(1157, '0x485', 0, 5, 5, 'storage_controller', 0.8, 'Group 0, Device 5: storage_controller', 1),
(1158, '0x486', 0, 6, 6, 'memory_controller', 0.9, 'Group 0, Device 6: memory_controller', 1),
(1159, '0x487', 0, 7, 7, 'display_controller', 0.7, 'Group 0, Device 7: display_controller', 1),
(1160, '0x488', 0, 8, 8, 'audio_controller', 0.6, 'Group 0, Device 8: audio_controller', 1),
(1161, '0x489', 0, 9, 9, 'usb_controller', 0.8, 'Group 0, Device 9: usb_controller', 1),
(1162, '0x48A', 0, 10, 10, 'expansion_slot', 0.5, 'Group 0, Device 10: expansion_slot', 1),
(1163, '0x48B', 0, 11, 11, 'maintenance_mode', 0.8, 'Group 0, Device 11: maintenance_mode', 1),

-- Group 1 (Tokens 1164-1175, 0x48C-0x497)
(1164, '0x48C', 1, 0, 12, 'power_management', 0.9, 'Group 1, Device 0: power_management', 0),
(1165, '0x48D', 1, 1, 13, 'thermal_control', 0.9, 'Group 1, Device 1: thermal_control', 1),
(1166, '0x48E', 1, 2, 14, 'security_module', 0.8, 'Group 1, Device 2: security_module', 1),
(1167, '0x48F', 1, 3, 15, 'diagnostic_mode', 0.7, 'Group 1, Device 3: diagnostic_mode', 1),
(1168, '0x490', 1, 4, 16, 'network_interface', 0.8, 'Group 1, Device 4: network_interface', 1),
(1169, '0x491', 1, 5, 17, 'storage_controller', 0.8, 'Group 1, Device 5: storage_controller', 1),
(1170, '0x492', 1, 6, 18, 'memory_controller', 0.9, 'Group 1, Device 6: memory_controller', 1),
(1171, '0x493', 1, 7, 19, 'display_controller', 0.7, 'Group 1, Device 7: display_controller', 1),
(1172, '0x494', 1, 8, 20, 'audio_controller', 0.6, 'Group 1, Device 8: audio_controller', 1),
(1173, '0x495', 1, 9, 21, 'usb_controller', 0.8, 'Group 1, Device 9: usb_controller', 1),
(1174, '0x496', 1, 10, 22, 'expansion_slot', 0.5, 'Group 1, Device 10: expansion_slot', 1),
(1175, '0x497', 1, 11, 23, 'maintenance_mode', 0.8, 'Group 1, Device 11: maintenance_mode', 1),

-- Group 2 (Tokens 1176-1187, 0x498-0x4A3)
(1176, '0x498', 2, 0, 24, 'power_management', 0.9, 'Group 2, Device 0: power_management', 0),
(1177, '0x499', 2, 1, 25, 'thermal_control', 0.9, 'Group 2, Device 1: thermal_control', 1),
(1178, '0x49A', 2, 2, 26, 'security_module', 0.8, 'Group 2, Device 2: security_module', 1),
(1179, '0x49B', 2, 3, 27, 'diagnostic_mode', 0.7, 'Group 2, Device 3: diagnostic_mode', 1),
(1180, '0x49C', 2, 4, 28, 'network_interface', 0.8, 'Group 2, Device 4: network_interface', 1),
(1181, '0x49D', 2, 5, 29, 'storage_controller', 0.8, 'Group 2, Device 5: storage_controller', 1),
(1182, '0x49E', 2, 6, 30, 'memory_controller', 0.9, 'Group 2, Device 6: memory_controller', 1),
(1183, '0x49F', 2, 7, 31, 'display_controller', 0.7, 'Group 2, Device 7: display_controller', 1),
(1184, '0x4A0', 2, 8, 32, 'audio_controller', 0.6, 'Group 2, Device 8: audio_controller', 1),
(1185, '0x4A1', 2, 9, 33, 'usb_controller', 0.8, 'Group 2, Device 9: usb_controller', 1),
(1186, '0x4A2', 2, 10, 34, 'expansion_slot', 0.5, 'Group 2, Device 10: expansion_slot', 1),
(1187, '0x4A3', 2, 11, 35, 'maintenance_mode', 0.8, 'Group 2, Device 11: maintenance_mode', 1),

-- Group 3 (Tokens 1188-1199, 0x4A4-0x4AF)
(1188, '0x4A4', 3, 0, 36, 'power_management', 0.9, 'Group 3, Device 0: power_management', 0),
(1189, '0x4A5', 3, 1, 37, 'thermal_control', 0.9, 'Group 3, Device 1: thermal_control', 1),
(1190, '0x4A6', 3, 2, 38, 'security_module', 0.8, 'Group 3, Device 2: security_module', 1),
(1191, '0x4A7', 3, 3, 39, 'diagnostic_mode', 0.7, 'Group 3, Device 3: diagnostic_mode', 1),
(1192, '0x4A8', 3, 4, 40, 'network_interface', 0.8, 'Group 3, Device 4: network_interface', 1),
(1193, '0x4A9', 3, 5, 41, 'storage_controller', 0.8, 'Group 3, Device 5: storage_controller', 1),
(1194, '0x4AA', 3, 6, 42, 'memory_controller', 0.9, 'Group 3, Device 6: memory_controller', 1),
(1195, '0x4AB', 3, 7, 43, 'display_controller', 0.7, 'Group 3, Device 7: display_controller', 1),
(1196, '0x4AC', 3, 8, 44, 'audio_controller', 0.6, 'Group 3, Device 8: audio_controller', 1),
(1197, '0x4AD', 3, 9, 45, 'usb_controller', 0.8, 'Group 3, Device 9: usb_controller', 1),
(1198, '0x4AE', 3, 10, 46, 'expansion_slot', 0.5, 'Group 3, Device 10: expansion_slot', 1),
(1199, '0x4AF', 3, 11, 47, 'maintenance_mode', 0.8, 'Group 3, Device 11: maintenance_mode', 1),

-- Group 4 (Tokens 1200-1211, 0x4B0-0x4BB)
(1200, '0x4B0', 4, 0, 48, 'power_management', 0.9, 'Group 4, Device 0: power_management', 0),
(1201, '0x4B1', 4, 1, 49, 'thermal_control', 0.9, 'Group 4, Device 1: thermal_control', 1),
(1202, '0x4B2', 4, 2, 50, 'security_module', 0.8, 'Group 4, Device 2: security_module', 1),
(1203, '0x4B3', 4, 3, 51, 'diagnostic_mode', 0.7, 'Group 4, Device 3: diagnostic_mode', 1),
(1204, '0x4B4', 4, 4, 52, 'network_interface', 0.8, 'Group 4, Device 4: network_interface', 1),
(1205, '0x4B5', 4, 5, 53, 'storage_controller', 0.8, 'Group 4, Device 5: storage_controller', 1),
(1206, '0x4B6', 4, 6, 54, 'memory_controller', 0.9, 'Group 4, Device 6: memory_controller', 1),
(1207, '0x4B7', 4, 7, 55, 'display_controller', 0.7, 'Group 4, Device 7: display_controller', 1),
(1208, '0x4B8', 4, 8, 56, 'audio_controller', 0.6, 'Group 4, Device 8: audio_controller', 1),
(1209, '0x4B9', 4, 9, 57, 'usb_controller', 0.8, 'Group 4, Device 9: usb_controller', 1),
(1210, '0x4BA', 4, 10, 58, 'expansion_slot', 0.5, 'Group 4, Device 10: expansion_slot', 1),
(1211, '0x4BB', 4, 11, 59, 'maintenance_mode', 0.8, 'Group 4, Device 11: maintenance_mode', 1),

-- Group 5 (Tokens 1212-1223, 0x4BC-0x4C7)
(1212, '0x4BC', 5, 0, 60, 'power_management', 0.9, 'Group 5, Device 0: power_management', 0),
(1213, '0x4BD', 5, 1, 61, 'thermal_control', 0.9, 'Group 5, Device 1: thermal_control', 1),
(1214, '0x4BE', 5, 2, 62, 'security_module', 0.8, 'Group 5, Device 2: security_module', 1),
(1215, '0x4BF', 5, 3, 63, 'diagnostic_mode', 0.7, 'Group 5, Device 3: diagnostic_mode', 1),
(1216, '0x4C0', 5, 4, 64, 'network_interface', 0.8, 'Group 5, Device 4: network_interface', 1),
(1217, '0x4C1', 5, 5, 65, 'storage_controller', 0.8, 'Group 5, Device 5: storage_controller', 1),
(1218, '0x4C2', 5, 6, 66, 'memory_controller', 0.9, 'Group 5, Device 6: memory_controller', 1),
(1219, '0x4C3', 5, 7, 67, 'display_controller', 0.7, 'Group 5, Device 7: display_controller', 1),
(1220, '0x4C4', 5, 8, 68, 'audio_controller', 0.6, 'Group 5, Device 8: audio_controller', 1),
(1221, '0x4C5', 5, 9, 69, 'usb_controller', 0.8, 'Group 5, Device 9: usb_controller', 1),
(1222, '0x4C6', 5, 10, 70, 'expansion_slot', 0.5, 'Group 5, Device 10: expansion_slot', 1),
(1223, '0x4C7', 5, 11, 71, 'maintenance_mode', 0.8, 'Group 5, Device 11: maintenance_mode', 1);

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================