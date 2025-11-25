#!/usr/bin/env python3
"""
DSMIL Audit Storage - Persistent Audit Event Storage
Compliance-ready audit logging with SQLite backend

Features:
- Persistent storage of all device operations
- Indexed queries by timestamp, device, operation
- Risk level tracking
- User attribution
- Time-range filtering
- Retention policy support
- Export capabilities
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Audit event risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DSMILAuditStorage:
    """Persistent storage for DSMIL audit events"""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize audit storage

        Args:
            db_path: Path to SQLite database (default: /var/lib/dsmil/audit.db)
        """
        if db_path is None:
            # Try system path first, fall back to user path
            system_path = Path("/var/lib/dsmil")
            if system_path.exists() and os.access(system_path, os.W_OK):
                self.db_path = system_path / "audit.db"
            else:
                # Fall back to user home directory
                user_path = Path.home() / ".dsmil"
                user_path.mkdir(parents=True, exist_ok=True)
                self.db_path = user_path / "audit.db"
        else:
            self.db_path = Path(db_path)

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"Audit storage initialized at {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Main audit events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                datetime_iso TEXT NOT NULL,
                device_id TEXT NOT NULL,
                device_name TEXT,
                operation TEXT NOT NULL,
                user TEXT,
                success BOOLEAN NOT NULL,
                details TEXT,
                value INTEGER,
                risk_level TEXT NOT NULL DEFAULT 'low',
                session_id TEXT,
                thermal_impact REAL,
                rollback_available BOOLEAN DEFAULT 0
            )
        ''')

        # Create indexes for fast querying
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON audit_events(timestamp DESC)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_device
            ON audit_events(device_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_operation
            ON audit_events(operation)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_success
            ON audit_events(success)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_risk_level
            ON audit_events(risk_level)
        ''')

        # Summary statistics table (for fast dashboard queries)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_stats (
                stat_date TEXT PRIMARY KEY,
                total_events INTEGER DEFAULT 0,
                successful_events INTEGER DEFAULT 0,
                failed_events INTEGER DEFAULT 0,
                critical_events INTEGER DEFAULT 0,
                unique_devices INTEGER DEFAULT 0,
                last_updated REAL
            )
        ''')

        conn.commit()
        conn.close()

        logger.info(f"Audit database schema initialized")

    def store_event(
        self,
        device_id: int,
        operation: str,
        success: bool,
        device_name: Optional[str] = None,
        user: Optional[str] = None,
        details: str = "",
        value: Optional[int] = None,
        risk_level: RiskLevel = RiskLevel.LOW,
        session_id: Optional[str] = None,
        thermal_impact: Optional[float] = None,
        rollback_available: bool = False
    ) -> int:
        """
        Store audit event to database

        Args:
            device_id: DSMIL device ID
            operation: Operation type (e.g., 'activate', 'read', 'write')
            success: Whether operation succeeded
            device_name: Human-readable device name
            user: Username performing operation
            details: Additional details or error message
            value: Optional value associated with operation
            risk_level: Risk level of operation
            session_id: Session identifier for grouping operations
            thermal_impact: Temperature change from operation (°C)
            rollback_available: Whether rollback is possible

        Returns:
            Event ID in database
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        now = datetime.now()
        timestamp = now.timestamp()
        datetime_iso = now.isoformat()

        # Get current user if not specified
        if user is None:
            user = os.environ.get('USER', 'unknown')

        cursor.execute('''
            INSERT INTO audit_events
            (timestamp, datetime_iso, device_id, device_name, operation, user,
             success, details, value, risk_level, session_id, thermal_impact,
             rollback_available)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp,
            datetime_iso,
            f"0x{device_id:04X}",
            device_name,
            operation,
            user,
            success,
            details,
            value,
            risk_level.value if isinstance(risk_level, RiskLevel) else risk_level,
            session_id,
            thermal_impact,
            rollback_available
        ))

        event_id = cursor.lastrowid

        # Update daily statistics
        stat_date = now.strftime('%Y-%m-%d')
        cursor.execute('''
            INSERT INTO audit_stats (stat_date, total_events, successful_events,
                                    failed_events, critical_events, last_updated)
            VALUES (?, 1, ?, ?, ?, ?)
            ON CONFLICT(stat_date) DO UPDATE SET
                total_events = total_events + 1,
                successful_events = successful_events + ?,
                failed_events = failed_events + ?,
                critical_events = critical_events + ?,
                last_updated = ?
        ''', (
            stat_date,
            1 if success else 0,
            0 if success else 1,
            1 if risk_level in [RiskLevel.CRITICAL, 'critical'] else 0,
            now.timestamp(),
            1 if success else 0,
            0 if success else 1,
            1 if risk_level in [RiskLevel.CRITICAL, 'critical'] else 0,
            now.timestamp()
        ))

        conn.commit()
        conn.close()

        logger.debug(f"Stored audit event {event_id}: {operation} on {device_id:04X}")
        return event_id

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        device_id: Optional[int] = None,
        operation: Optional[str] = None,
        success: Optional[bool] = None,
        risk_level: Optional[RiskLevel] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        user: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Query audit events with filtering

        Args:
            limit: Maximum number of events to return
            offset: Number of events to skip
            device_id: Filter by device ID
            operation: Filter by operation type
            success: Filter by success status
            risk_level: Filter by risk level
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            user: Filter by username
            session_id: Filter by session ID

        Returns:
            List of audit events as dictionaries
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []

        if device_id is not None:
            query += " AND device_id = ?"
            params.append(f"0x{device_id:04X}")

        if operation:
            query += " AND operation = ?"
            params.append(operation)

        if success is not None:
            query += " AND success = ?"
            params.append(success)

        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level.value if isinstance(risk_level, RiskLevel) else risk_level)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        if user:
            query += " AND user = ?"
            params.append(user)

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)

        columns = [desc[0] for desc in cursor.description]
        events = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return events

    def get_statistics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Get audit statistics summary

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Overall statistics
        query = "SELECT COUNT(*), SUM(success), SUM(NOT success) FROM audit_events"
        params = []

        if start_date or end_date:
            query += " WHERE 1=1"
            if start_date:
                start_ts = datetime.strptime(start_date, '%Y-%m-%d').timestamp()
                query += " AND timestamp >= ?"
                params.append(start_ts)
            if end_date:
                end_ts = datetime.strptime(end_date, '%Y-%m-%d').timestamp()
                query += " AND timestamp <= ?"
                params.append(end_ts)

        cursor.execute(query, params)
        total, successful, failed = cursor.fetchone()

        # Operations by type
        cursor.execute('''
            SELECT operation, COUNT(*) as count
            FROM audit_events
            GROUP BY operation
            ORDER BY count DESC
        ''')
        operations = {row[0]: row[1] for row in cursor.fetchall()}

        # Most active devices
        cursor.execute('''
            SELECT device_id, device_name, COUNT(*) as count
            FROM audit_events
            GROUP BY device_id
            ORDER BY count DESC
            LIMIT 10
        ''')
        active_devices = [
            {'device_id': row[0], 'device_name': row[1], 'count': row[2]}
            for row in cursor.fetchall()
        ]

        # Risk level breakdown
        cursor.execute('''
            SELECT risk_level, COUNT(*) as count
            FROM audit_events
            GROUP BY risk_level
        ''')
        risk_levels = {row[0]: row[1] for row in cursor.fetchall()}

        # Recent critical events
        cursor.execute('''
            SELECT datetime_iso, device_id, operation, details
            FROM audit_events
            WHERE risk_level = 'critical'
            ORDER BY timestamp DESC
            LIMIT 5
        ''')
        critical_events = [
            {
                'timestamp': row[0],
                'device_id': row[1],
                'operation': row[2],
                'details': row[3]
            }
            for row in cursor.fetchall()
        ]

        conn.close()

        return {
            'total_events': total or 0,
            'successful_events': successful or 0,
            'failed_events': failed or 0,
            'success_rate': round((successful / total * 100) if total > 0 else 0, 2),
            'operations_by_type': operations,
            'most_active_devices': active_devices,
            'risk_level_breakdown': risk_levels,
            'recent_critical_events': critical_events
        }

    def export_events(
        self,
        output_path: Path,
        format: str = 'json',
        **filters
    ) -> int:
        """
        Export audit events to file

        Args:
            output_path: Path for output file
            format: Export format ('json', 'csv', 'html')
            **filters: Same filters as get_events()

        Returns:
            Number of events exported
        """
        events = self.get_events(limit=100000, **filters)  # High limit for export

        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(events, f, indent=2)

        elif format == 'csv':
            import csv
            if events:
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=events[0].keys())
                    writer.writeheader()
                    writer.writerows(events)

        elif format == 'html':
            html = self._generate_html_report(events)
            with open(output_path, 'w') as f:
                f.write(html)

        logger.info(f"Exported {len(events)} events to {output_path}")
        return len(events)

    def _generate_html_report(self, events: List[Dict]) -> str:
        """Generate HTML audit report"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>DSMIL Audit Report</title>
    <style>
        body { font-family: monospace; background: #000; color: #0f0; padding: 20px; }
        h1 { color: #ff0; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #0f0; padding: 8px; text-align: left; }
        th { background: #003300; color: #ff0; }
        .success { color: #0f0; }
        .failure { color: #f00; }
        .critical { background: #330000; }
    </style>
</head>
<body>
    <h1>DSMIL Audit Report</h1>
    <p>Generated: {datetime}</p>
    <p>Total Events: {count}</p>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Device</th>
            <th>Operation</th>
            <th>User</th>
            <th>Status</th>
            <th>Details</th>
            <th>Risk</th>
        </tr>
""".format(datetime=datetime.now().isoformat(), count=len(events))

        for event in events:
            status_class = 'success' if event['success'] else 'failure'
            risk_class = 'critical' if event['risk_level'] == 'critical' else ''

            html += f"""        <tr class="{risk_class}">
            <td>{event['datetime_iso']}</td>
            <td>{event['device_name'] or event['device_id']}</td>
            <td>{event['operation']}</td>
            <td>{event['user']}</td>
            <td class="{status_class}">{'✓' if event['success'] else '✗'}</td>
            <td>{event['details']}</td>
            <td>{event['risk_level']}</td>
        </tr>
"""

        html += """    </table>
</body>
</html>"""
        return html

    def cleanup_old_events(self, retention_days: int = 90) -> int:
        """
        Remove audit events older than retention period

        Args:
            retention_days: Number of days to retain events

        Returns:
            Number of events deleted
        """
        cutoff_time = (datetime.now() - timedelta(days=retention_days)).timestamp()

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM audit_events WHERE timestamp < ?', (cutoff_time,))
        count = cursor.fetchone()[0]

        cursor.execute('DELETE FROM audit_events WHERE timestamp < ?', (cutoff_time,))

        # Also cleanup old stats
        cutoff_date = (datetime.now() - timedelta(days=retention_days)).strftime('%Y-%m-%d')
        cursor.execute('DELETE FROM audit_stats WHERE stat_date < ?', (cutoff_date,))

        conn.commit()
        conn.close()

        logger.info(f"Cleaned up {count} audit events older than {retention_days} days")
        return count

    def get_database_size(self) -> Dict[str, any]:
        """Get audit database size and statistics"""
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM audit_events')
        event_count = cursor.fetchone()[0]

        cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM audit_events')
        min_ts, max_ts = cursor.fetchone()

        conn.close()

        return {
            'database_path': str(self.db_path),
            'size_bytes': size_bytes,
            'size_mb': round(size_bytes / (1024 * 1024), 2),
            'event_count': event_count,
            'oldest_event': datetime.fromtimestamp(min_ts).isoformat() if min_ts else None,
            'newest_event': datetime.fromtimestamp(max_ts).isoformat() if max_ts else None,
            'date_range_days': round((max_ts - min_ts) / 86400, 1) if min_ts and max_ts else 0
        }
