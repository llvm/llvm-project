#!/usr/bin/env python3
"""
DSMIL Token Testing Database Backend System
Multi-format data storage with SQLite, JSON, CSV, and binary backends
Version: 1.0.0
Date: 2025-09-01
"""

import sqlite3
import json
import csv
import pickle
import struct
import os
import time
import uuid
import hashlib
import threading
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/home/john/LAT5150DRVMIL/database/backends/database_backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TokenTestResult:
    """Data structure for individual token test results"""
    test_id: str
    session_id: str
    token_id: int
    hex_id: str
    group_id: int
    device_id: int
    test_timestamp: float
    access_method: str
    operation_type: str
    initial_value: Optional[str] = None
    set_value: Optional[str] = None
    final_value: Optional[str] = None
    expected_value: Optional[str] = None
    test_duration_ms: Optional[int] = None
    success: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    notes: Optional[str] = None

@dataclass
class ThermalReading:
    """Thermal sensor reading data"""
    reading_id: str
    test_id: Optional[str]
    session_id: str
    reading_timestamp: float
    sensor_name: str
    temperature_celsius: float
    critical_temp: Optional[float] = None
    warning_temp: Optional[float] = None
    thermal_state: str = "normal"
    fan_speed_rpm: Optional[int] = None
    thermal_throttling: bool = False

@dataclass
class SystemMetric:
    """System performance metrics"""
    metric_id: str
    test_id: Optional[str]
    session_id: str
    metric_timestamp: float
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_available_gb: Optional[float] = None
    disk_usage_percent: Optional[float] = None
    system_load_1min: Optional[float] = None
    system_load_5min: Optional[float] = None
    system_load_15min: Optional[float] = None
    uptime_hours: Optional[float] = None
    process_count: Optional[int] = None

@dataclass
class DSMILResponse:
    """DSMIL device response data"""
    response_id: str
    test_id: Optional[str]
    session_id: str
    response_timestamp: float
    group_id: int
    device_id: Optional[int]
    response_type: str
    previous_state: Optional[str] = None
    new_state: Optional[str] = None
    response_data: Optional[str] = None
    memory_address: Optional[str] = None
    memory_size: Optional[int] = None
    correlation_strength: Optional[float] = None
    response_delay_ms: Optional[int] = None

class DatabaseBackend:
    """Comprehensive multi-format database backend for DSMIL token testing"""
    
    def __init__(self, base_path: str = "/home/john/LAT5150DRVMIL/database"):
        self.base_path = Path(base_path)
        self.sqlite_db = self.base_path / "data" / "dsmil_tokens.db"
        self.json_dir = self.base_path / "data" / "json"
        self.csv_dir = self.base_path / "data" / "csv"
        self.binary_dir = self.base_path / "data" / "binary"
        self.backup_dir = self.base_path / "backups"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize all storage backends
        self._initialize_storage()
        
    def _initialize_storage(self):
        """Initialize all storage backend directories and files"""
        # Create directories
        for directory in [self.sqlite_db.parent, self.json_dir, self.csv_dir, 
                         self.binary_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Initialize SQLite database
        self._initialize_sqlite()
        
        # Initialize CSV headers
        self._initialize_csv_files()
        
        logger.info(f"Database backend initialized at {self.base_path}")
        
    def _initialize_sqlite(self):
        """Initialize SQLite database with schema"""
        schema_file = self.base_path / "schemas" / "dsmil_tokens.sql"
        
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            return
            
        with sqlite3.connect(str(self.sqlite_db)) as conn:
            # Check if database is already initialized
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test_sessions'")
            
            if not cursor.fetchone():
                # Database not initialized, execute schema
                with open(schema_file, 'r') as f:
                    schema_sql = f.read()
                conn.executescript(schema_sql)
                logger.info("SQLite database schema initialized")
            else:
                logger.info("SQLite database already initialized")
                
    def _initialize_csv_files(self):
        """Initialize CSV files with headers"""
        csv_files = {
            'test_sessions.csv': [
                'session_id', 'session_name', 'start_timestamp', 'end_timestamp',
                'session_type', 'total_tokens', 'successful_tokens', 'failed_tokens',
                'emergency_stops', 'thermal_warnings', 'status', 'operator', 'notes'
            ],
            'token_tests.csv': [
                'test_id', 'session_id', 'token_id', 'test_timestamp', 'access_method',
                'operation_type', 'initial_value', 'set_value', 'final_value',
                'expected_value', 'test_duration_ms', 'success', 'error_code',
                'error_message', 'retry_count', 'notes'
            ],
            'thermal_readings.csv': [
                'reading_id', 'test_id', 'session_id', 'reading_timestamp',
                'sensor_name', 'temperature_celsius', 'critical_temp', 'warning_temp',
                'thermal_state', 'fan_speed_rpm', 'thermal_throttling'
            ],
            'system_metrics.csv': [
                'metric_id', 'test_id', 'session_id', 'metric_timestamp',
                'cpu_percent', 'memory_percent', 'memory_available_gb',
                'disk_usage_percent', 'system_load_1min', 'system_load_5min',
                'system_load_15min', 'uptime_hours', 'process_count'
            ],
            'dsmil_responses.csv': [
                'response_id', 'test_id', 'session_id', 'response_timestamp',
                'group_id', 'device_id', 'response_type', 'previous_state',
                'new_state', 'response_data', 'memory_address', 'memory_size',
                'correlation_strength', 'response_delay_ms'
            ]
        }
        
        for filename, headers in csv_files.items():
            csv_path = self.csv_dir / filename
            if not csv_path.exists():
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    
    @contextmanager
    def _get_sqlite_connection(self):
        """Thread-safe SQLite connection context manager"""
        with self._lock:
            conn = sqlite3.connect(str(self.sqlite_db))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
                
    def create_session(self, session_name: str, session_type: str, 
                      operator: Optional[str] = None) -> str:
        """Create a new test session"""
        session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        timestamp = time.time()
        
        session_data = {
            'session_id': session_id,
            'session_name': session_name,
            'start_timestamp': timestamp,
            'session_type': session_type,
            'status': 'running',
            'operator': operator,
            'total_tokens': 0,
            'successful_tokens': 0,
            'failed_tokens': 0,
            'emergency_stops': 0,
            'thermal_warnings': 0
        }
        
        # Store in SQLite
        with self._get_sqlite_connection() as conn:
            conn.execute("""
                INSERT INTO test_sessions (
                    session_id, session_name, start_timestamp, session_type,
                    status, operator, total_tokens, successful_tokens,
                    failed_tokens, emergency_stops, thermal_warnings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, session_name, timestamp, session_type,
                'running', operator, 0, 0, 0, 0, 0
            ))
            conn.commit()
            
        # Store in JSON
        json_file = self.json_dir / f"{session_id}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'session_info': session_data,
                'token_tests': [],
                'thermal_readings': [],
                'system_metrics': [],
                'dsmil_responses': []
            }, f, indent=2)
            
        # Store in CSV
        csv_file = self.csv_dir / "test_sessions.csv"
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=session_data.keys())
            writer.writerow(session_data)
            
        # Create binary session file
        binary_file = self.binary_dir / f"{session_id}.dsm"
        with open(binary_file, 'wb') as f:
            # Binary header: magic, version, session_id_len, session_id
            f.write(b'DSML')  # Magic number
            f.write(struct.pack('<I', 1))  # Version
            f.write(struct.pack('<I', len(session_id)))
            f.write(session_id.encode('utf-8'))
            f.write(struct.pack('<d', timestamp))  # Start timestamp
            
        logger.info(f"Created session {session_id}: {session_name}")
        return session_id
        
    def record_token_test(self, result: TokenTestResult):
        """Record a token test result in all backends"""
        with self._lock:
            # Store in SQLite
            with self._get_sqlite_connection() as conn:
                conn.execute("""
                    INSERT INTO token_tests (
                        test_id, session_id, token_id, test_timestamp, access_method,
                        operation_type, initial_value, set_value, final_value,
                        expected_value, test_duration_ms, success, error_code,
                        error_message, retry_count, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.test_id, result.session_id, result.token_id,
                    result.test_timestamp, result.access_method, result.operation_type,
                    result.initial_value, result.set_value, result.final_value,
                    result.expected_value, result.test_duration_ms, result.success,
                    result.error_code, result.error_message, result.retry_count,
                    result.notes
                ))
                conn.commit()
                
            # Update JSON session file
            json_file = self.json_dir / f"{result.session_id}.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    session_data = json.load(f)
                session_data['token_tests'].append(asdict(result))
                with open(json_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                    
            # Append to CSV
            csv_file = self.csv_dir / "token_tests.csv"
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(result).keys())
                writer.writerow(asdict(result))
                
            # Append to binary file
            binary_file = self.binary_dir / f"{result.session_id}.dsm"
            with open(binary_file, 'ab') as f:
                # Record type marker
                f.write(b'TEST')
                # Serialize test result
                data = pickle.dumps(result)
                f.write(struct.pack('<I', len(data)))
                f.write(data)
                
            logger.debug(f"Recorded token test {result.test_id}")
            
    def record_thermal_reading(self, reading: ThermalReading):
        """Record a thermal reading in all backends"""
        with self._lock:
            # Store in SQLite
            with self._get_sqlite_connection() as conn:
                conn.execute("""
                    INSERT INTO thermal_readings (
                        reading_id, test_id, session_id, reading_timestamp,
                        sensor_name, temperature_celsius, critical_temp, warning_temp,
                        thermal_state, fan_speed_rpm, thermal_throttling
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    reading.reading_id, reading.test_id, reading.session_id,
                    reading.reading_timestamp, reading.sensor_name,
                    reading.temperature_celsius, reading.critical_temp,
                    reading.warning_temp, reading.thermal_state,
                    reading.fan_speed_rpm, reading.thermal_throttling
                ))
                conn.commit()
                
            # Update JSON
            json_file = self.json_dir / f"{reading.session_id}.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    session_data = json.load(f)
                session_data['thermal_readings'].append(asdict(reading))
                with open(json_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                    
            # Append to CSV
            csv_file = self.csv_dir / "thermal_readings.csv"
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(reading).keys())
                writer.writerow(asdict(reading))
                
            # Binary storage
            binary_file = self.binary_dir / f"{reading.session_id}.dsm"
            with open(binary_file, 'ab') as f:
                f.write(b'THRM')
                data = pickle.dumps(reading)
                f.write(struct.pack('<I', len(data)))
                f.write(data)
                
    def record_system_metric(self, metric: SystemMetric):
        """Record a system metric in all backends"""
        with self._lock:
            # Store in SQLite
            with self._get_sqlite_connection() as conn:
                conn.execute("""
                    INSERT INTO system_metrics (
                        metric_id, test_id, session_id, metric_timestamp,
                        cpu_percent, memory_percent, memory_available_gb,
                        disk_usage_percent, system_load_1min, system_load_5min,
                        system_load_15min, uptime_hours, process_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.metric_id, metric.test_id, metric.session_id,
                    metric.metric_timestamp, metric.cpu_percent, metric.memory_percent,
                    metric.memory_available_gb, metric.disk_usage_percent,
                    metric.system_load_1min, metric.system_load_5min,
                    metric.system_load_15min, metric.uptime_hours, metric.process_count
                ))
                conn.commit()
                
            # Update JSON
            json_file = self.json_dir / f"{metric.session_id}.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    session_data = json.load(f)
                session_data['system_metrics'].append(asdict(metric))
                with open(json_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                    
            # CSV storage
            csv_file = self.csv_dir / "system_metrics.csv"
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(metric).keys())
                writer.writerow(asdict(metric))
                
            # Binary storage
            binary_file = self.binary_dir / f"{metric.session_id}.dsm"
            with open(binary_file, 'ab') as f:
                f.write(b'SYMT')
                data = pickle.dumps(metric)
                f.write(struct.pack('<I', len(data)))
                f.write(data)
                
    def record_dsmil_response(self, response: DSMILResponse):
        """Record a DSMIL device response in all backends"""
        with self._lock:
            # Store in SQLite
            with self._get_sqlite_connection() as conn:
                conn.execute("""
                    INSERT INTO dsmil_responses (
                        response_id, test_id, session_id, response_timestamp,
                        group_id, device_id, response_type, previous_state,
                        new_state, response_data, memory_address, memory_size,
                        correlation_strength, response_delay_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    response.response_id, response.test_id, response.session_id,
                    response.response_timestamp, response.group_id, response.device_id,
                    response.response_type, response.previous_state, response.new_state,
                    response.response_data, response.memory_address, response.memory_size,
                    response.correlation_strength, response.response_delay_ms
                ))
                conn.commit()
                
            # Update JSON
            json_file = self.json_dir / f"{response.session_id}.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    session_data = json.load(f)
                session_data['dsmil_responses'].append(asdict(response))
                with open(json_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                    
            # CSV storage
            csv_file = self.csv_dir / "dsmil_responses.csv"
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(response).keys())
                writer.writerow(asdict(response))
                
            # Binary storage
            binary_file = self.binary_dir / f"{response.session_id}.dsm"
            with open(binary_file, 'ab') as f:
                f.write(b'DSML')
                data = pickle.dumps(response)
                f.write(struct.pack('<I', len(data)))
                f.write(data)
                
    def close_session(self, session_id: str, status: str = "completed", notes: Optional[str] = None):
        """Close a test session"""
        end_timestamp = time.time()
        
        with self._get_sqlite_connection() as conn:
            conn.execute("""
                UPDATE test_sessions 
                SET end_timestamp = ?, status = ?, notes = ?, updated_at = julianday('now')
                WHERE session_id = ?
            """, (end_timestamp, status, notes, session_id))
            conn.commit()
            
        logger.info(f"Closed session {session_id} with status: {status}")
        
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a comprehensive backup of all data"""
        if not backup_name:
            backup_name = f"backup_{int(time.time())}"
            
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # Create manifest
        manifest = {
            'backup_name': backup_name,
            'timestamp': time.time(),
            'files': {}
        }
        
        # Copy SQLite database
        import shutil
        if self.sqlite_db.exists():
            dest = backup_path / "dsmil_tokens.db"
            shutil.copy2(self.sqlite_db, dest)
            manifest['files']['sqlite'] = str(dest.name)
            
        # Archive JSON files
        json_archive = backup_path / "json_data.tar.gz"
        if self.json_dir.exists():
            import tarfile
            with tarfile.open(json_archive, 'w:gz') as tar:
                tar.add(self.json_dir, arcname='json')
            manifest['files']['json'] = str(json_archive.name)
            
        # Archive CSV files
        csv_archive = backup_path / "csv_data.tar.gz"
        if self.csv_dir.exists():
            with tarfile.open(csv_archive, 'w:gz') as tar:
                tar.add(self.csv_dir, arcname='csv')
            manifest['files']['csv'] = str(csv_archive.name)
            
        # Archive binary files
        binary_archive = backup_path / "binary_data.tar.gz"
        if self.binary_dir.exists():
            with tarfile.open(binary_archive, 'w:gz') as tar:
                tar.add(self.binary_dir, arcname='binary')
            manifest['files']['binary'] = str(binary_archive.name)
            
        # Save manifest
        with open(backup_path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
            
        logger.info(f"Created backup: {backup_name}")
        return backup_name
        
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get session summary with statistics"""
        with self._get_sqlite_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM session_summary WHERE session_id = ?
            """, (session_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
    def query_token_tests(self, session_id: Optional[str] = None, 
                         token_id: Optional[int] = None,
                         success: Optional[bool] = None,
                         limit: Optional[int] = None) -> List[Dict]:
        """Query token test results with filters"""
        query = "SELECT * FROM token_tests WHERE 1=1"
        params = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
            
        if token_id:
            query += " AND token_id = ?"
            params.append(token_id)
            
        if success is not None:
            query += " AND success = ?"
            params.append(success)
            
        query += " ORDER BY test_timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
            
        with self._get_sqlite_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
    def get_thermal_analysis(self, session_id: str) -> Dict:
        """Get thermal analysis for a session"""
        with self._get_sqlite_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    sensor_name,
                    MIN(temperature_celsius) as min_temp,
                    MAX(temperature_celsius) as max_temp,
                    AVG(temperature_celsius) as avg_temp,
                    COUNT(*) as reading_count,
                    SUM(CASE WHEN thermal_state != 'normal' THEN 1 ELSE 0 END) as thermal_events
                FROM thermal_readings 
                WHERE session_id = ?
                GROUP BY sensor_name
            """, (session_id,))
            
            return {row['sensor_name']: dict(row) for row in cursor.fetchall()}
            
    def export_session_data(self, session_id: str, format_type: str = "json") -> str:
        """Export session data in specified format"""
        if format_type == "json":
            json_file = self.json_dir / f"{session_id}.json"
            if json_file.exists():
                return str(json_file)
                
        elif format_type == "csv":
            # Create session-specific CSV export
            export_path = self.csv_dir / f"session_{session_id}_export.csv"
            
            with self._get_sqlite_connection() as conn:
                cursor = conn.execute("""
                    SELECT 
                        tt.*,
                        td.hex_id,
                        td.group_id,
                        td.device_id,
                        td.potential_function
                    FROM token_tests tt
                    JOIN token_definitions td ON tt.token_id = td.token_id
                    WHERE tt.session_id = ?
                    ORDER BY tt.test_timestamp
                """, (session_id,))
                
                with open(export_path, 'w', newline='') as f:
                    if cursor.description:
                        fieldnames = [desc[0] for desc in cursor.description]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in cursor:
                            writer.writerow(dict(row))
                            
            return str(export_path)
            
        return ""
        
    def verify_data_integrity(self) -> Dict[str, Any]:
        """Verify data integrity across all backends"""
        results = {
            'timestamp': time.time(),
            'sqlite_integrity': False,
            'file_consistency': False,
            'backup_status': False,
            'issues': []
        }
        
        # Check SQLite integrity
        try:
            with self._get_sqlite_connection() as conn:
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                results['sqlite_integrity'] = (integrity_result == 'ok')
                if integrity_result != 'ok':
                    results['issues'].append(f"SQLite integrity: {integrity_result}")
        except Exception as e:
            results['issues'].append(f"SQLite integrity check failed: {str(e)}")
            
        # Check file consistency
        try:
            session_files = list(self.json_dir.glob("session_*.json"))
            binary_files = list(self.binary_dir.glob("session_*.dsm"))
            
            json_sessions = {f.stem for f in session_files}
            binary_sessions = {f.stem for f in binary_files}
            
            missing_binary = json_sessions - binary_sessions
            missing_json = binary_sessions - json_sessions
            
            if missing_binary or missing_json:
                results['issues'].append(f"File consistency: {len(missing_binary)} missing binary, {len(missing_json)} missing JSON")
            else:
                results['file_consistency'] = True
                
        except Exception as e:
            results['issues'].append(f"File consistency check failed: {str(e)}")
            
        # Check backup status
        try:
            backups = list(self.backup_dir.glob("backup_*"))
            results['backup_status'] = len(backups) > 0
            if not results['backup_status']:
                results['issues'].append("No backups found")
        except Exception as e:
            results['issues'].append(f"Backup check failed: {str(e)}")
            
        return results

if __name__ == "__main__":
    # Example usage and testing
    db = DatabaseBackend()
    
    # Test session creation
    session_id = db.create_session("Test Session", "single", "test_operator")
    print(f"Created session: {session_id}")
    
    # Test data recording
    test_result = TokenTestResult(
        test_id=f"test_{uuid.uuid4().hex[:8]}",
        session_id=session_id,
        token_id=1152,
        hex_id="0x480",
        group_id=0,
        device_id=0,
        test_timestamp=time.time(),
        access_method="smbios-token-ctl",
        operation_type="read",
        initial_value="0",
        success=True,
        test_duration_ms=250
    )
    
    db.record_token_test(test_result)
    print("Recorded test result")
    
    # Test integrity verification
    integrity = db.verify_data_integrity()
    print(f"Data integrity: {integrity}")
    
    # Create backup
    backup_name = db.create_backup()
    print(f"Created backup: {backup_name}")