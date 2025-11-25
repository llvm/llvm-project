#!/usr/bin/env python3
"""
DSMIL Database Integrity Management System
Atomic transactions, backup management, and data verification
Version: 1.0.0
Date: 2025-09-01
"""

import os
import sys
import sqlite3
import json
import hashlib
import shutil
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import tarfile
import pickle

# Add database backend to path
sys.path.insert(0, '/home/john/LAT5150DRVMIL/database/backends')
from database_backend import DatabaseBackend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('/home/john/LAT5150DRVMIL/database/tools/integrity_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class IntegrityCheck:
    """Integrity check result"""
    check_type: str
    status: str  # passed, failed, warning
    timestamp: float
    details: Dict[str, Any]
    recommendations: List[str]

@dataclass
class BackupInfo:
    """Backup information structure"""
    backup_id: str
    name: str
    timestamp: float
    size_bytes: int
    file_count: int
    checksum: str
    compression: str
    backup_path: str
    metadata: Dict[str, Any]

class TransactionManager:
    """Atomic transaction management for multi-backend operations"""
    
    def __init__(self, db_backend: DatabaseBackend):
        self.db = db_backend
        self._lock = threading.RLock()
        self._active_transactions = {}
        
    @contextmanager
    def transaction(self, transaction_id: str = None):
        """Context manager for atomic transactions across all backends"""
        if not transaction_id:
            transaction_id = f"txn_{int(time.time())}_{os.getpid()}"
            
        with self._lock:
            # Start transaction
            transaction_state = self._begin_transaction(transaction_id)
            
            try:
                yield transaction_state
                # Commit transaction
                self._commit_transaction(transaction_id, transaction_state)
                logger.info(f"Transaction {transaction_id} committed successfully")
                
            except Exception as e:
                # Rollback transaction
                self._rollback_transaction(transaction_id, transaction_state)
                logger.error(f"Transaction {transaction_id} rolled back due to: {str(e)}")
                raise
                
    def _begin_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Begin atomic transaction"""
        transaction_state = {
            'transaction_id': transaction_id,
            'start_time': time.time(),
            'sqlite_savepoint': None,
            'json_backups': {},
            'csv_positions': {},
            'binary_positions': {}
        }
        
        # Create SQLite savepoint
        with self.db._get_sqlite_connection() as conn:
            savepoint_name = f"sp_{transaction_id}"
            conn.execute(f"SAVEPOINT {savepoint_name}")
            transaction_state['sqlite_savepoint'] = savepoint_name
            
        # Record current file positions for rollback
        self._record_file_positions(transaction_state)
        
        self._active_transactions[transaction_id] = transaction_state
        return transaction_state
        
    def _commit_transaction(self, transaction_id: str, transaction_state: Dict[str, Any]):
        """Commit atomic transaction"""
        # Release SQLite savepoint
        with self.db._get_sqlite_connection() as conn:
            savepoint_name = transaction_state['sqlite_savepoint']
            conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
            conn.commit()
            
        # Clean up transaction state
        if transaction_id in self._active_transactions:
            del self._active_transactions[transaction_id]
            
    def _rollback_transaction(self, transaction_id: str, transaction_state: Dict[str, Any]):
        """Rollback atomic transaction"""
        try:
            # Rollback SQLite to savepoint
            with self.db._get_sqlite_connection() as conn:
                savepoint_name = transaction_state['sqlite_savepoint']
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                conn.commit()
                
            # Restore file positions
            self._restore_file_positions(transaction_state)
            
        except Exception as e:
            logger.error(f"Error during rollback of transaction {transaction_id}: {str(e)}")
            
        # Clean up transaction state
        if transaction_id in self._active_transactions:
            del self._active_transactions[transaction_id]
            
    def _record_file_positions(self, transaction_state: Dict[str, Any]):
        """Record current file positions for potential rollback"""
        # JSON file backups
        json_dir = self.db.json_dir
        if json_dir.exists():
            for json_file in json_dir.glob("*.json"):
                backup_path = json_file.with_suffix('.json.backup')
                shutil.copy2(json_file, backup_path)
                transaction_state['json_backups'][str(json_file)] = str(backup_path)
                
        # CSV file positions
        csv_dir = self.db.csv_dir
        if csv_dir.exists():
            for csv_file in csv_dir.glob("*.csv"):
                if csv_file.exists():
                    file_size = csv_file.stat().st_size
                    transaction_state['csv_positions'][str(csv_file)] = file_size
                    
        # Binary file positions
        binary_dir = self.db.binary_dir
        if binary_dir.exists():
            for binary_file in binary_dir.glob("*.dsm"):
                if binary_file.exists():
                    file_size = binary_file.stat().st_size
                    transaction_state['binary_positions'][str(binary_file)] = file_size
                    
    def _restore_file_positions(self, transaction_state: Dict[str, Any]):
        """Restore file positions for rollback"""
        # Restore JSON files
        for original_path, backup_path in transaction_state['json_backups'].items():
            if Path(backup_path).exists():
                shutil.move(backup_path, original_path)
                
        # Truncate CSV files to original positions
        for csv_path, original_size in transaction_state['csv_positions'].items():
            csv_file = Path(csv_path)
            if csv_file.exists() and csv_file.stat().st_size > original_size:
                with open(csv_file, 'r+b') as f:
                    f.truncate(original_size)
                    
        # Truncate binary files to original positions
        for binary_path, original_size in transaction_state['binary_positions'].items():
            binary_file = Path(binary_path)
            if binary_file.exists() and binary_file.stat().st_size > original_size:
                with open(binary_file, 'r+b') as f:
                    f.truncate(original_size)

class IntegrityManager:
    """Comprehensive data integrity management"""
    
    def __init__(self, db_backend: DatabaseBackend):
        self.db = db_backend
        self.transaction_manager = TransactionManager(db_backend)
        self.backup_dir = self.db.backup_dir
        self.verification_cache = {}
        
    def run_integrity_checks(self) -> List[IntegrityCheck]:
        """Run comprehensive integrity checks"""
        checks = []
        
        # SQLite integrity check
        checks.append(self._check_sqlite_integrity())
        
        # File consistency check
        checks.append(self._check_file_consistency())
        
        # Data consistency check
        checks.append(self._check_data_consistency())
        
        # Backup integrity check
        checks.append(self._check_backup_integrity())
        
        # Performance health check
        checks.append(self._check_performance_health())
        
        return checks
        
    def _check_sqlite_integrity(self) -> IntegrityCheck:
        """Check SQLite database integrity"""
        details = {}
        recommendations = []
        status = "passed"
        
        try:
            with self.db._get_sqlite_connection() as conn:
                # Pragma integrity check
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                details['integrity_check'] = integrity_result
                
                if integrity_result != 'ok':
                    status = "failed"
                    recommendations.append("SQLite database corruption detected. Run manual repair or restore from backup.")
                    
                # Check foreign key constraints
                cursor = conn.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                details['foreign_key_violations'] = len(fk_violations)
                
                if fk_violations:
                    status = "warning" if status == "passed" else status
                    recommendations.append(f"Found {len(fk_violations)} foreign key violations. Review data consistency.")
                    
                # Check table counts
                cursor = conn.execute("""
                    SELECT name, 
                           (SELECT COUNT(*) FROM sqlite_master sm WHERE sm.name = m.name) as count 
                    FROM sqlite_master m 
                    WHERE type = 'table'
                """)
                table_info = cursor.fetchall()
                details['table_count'] = len(table_info)
                
                # Check for missing indices
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
                index_count = len(cursor.fetchall())
                details['index_count'] = index_count
                
                if index_count < 10:  # Expected minimum indices
                    status = "warning" if status == "passed" else status
                    recommendations.append("Some database indices may be missing. This could impact performance.")
                    
        except Exception as e:
            status = "failed"
            details['error'] = str(e)
            recommendations.append("Unable to check SQLite integrity. Database may be corrupted or inaccessible.")
            
        return IntegrityCheck(
            check_type="sqlite_integrity",
            status=status,
            timestamp=time.time(),
            details=details,
            recommendations=recommendations
        )
        
    def _check_file_consistency(self) -> IntegrityCheck:
        """Check consistency between different storage backends"""
        details = {}
        recommendations = []
        status = "passed"
        
        try:
            # Get session lists from different backends
            json_sessions = set()
            binary_sessions = set()
            sqlite_sessions = set()
            
            # JSON sessions
            if self.db.json_dir.exists():
                for json_file in self.db.json_dir.glob("session_*.json"):
                    json_sessions.add(json_file.stem)
                    
            # Binary sessions
            if self.db.binary_dir.exists():
                for binary_file in self.db.binary_dir.glob("session_*.dsm"):
                    binary_sessions.add(binary_file.stem)
                    
            # SQLite sessions
            with self.db._get_sqlite_connection() as conn:
                cursor = conn.execute("SELECT session_id FROM test_sessions")
                sqlite_sessions = set(row[0] for row in cursor.fetchall())
                
            details['json_sessions'] = len(json_sessions)
            details['binary_sessions'] = len(binary_sessions)
            details['sqlite_sessions'] = len(sqlite_sessions)
            
            # Find inconsistencies
            missing_json = sqlite_sessions - json_sessions
            missing_binary = sqlite_sessions - binary_sessions
            orphan_json = json_sessions - sqlite_sessions
            orphan_binary = binary_sessions - sqlite_sessions
            
            details['missing_json'] = len(missing_json)
            details['missing_binary'] = len(missing_binary)
            details['orphan_json'] = len(orphan_json)
            details['orphan_binary'] = len(orphan_binary)
            
            if missing_json or missing_binary or orphan_json or orphan_binary:
                status = "warning"
                if missing_json:
                    recommendations.append(f"Found {len(missing_json)} sessions missing JSON files.")
                if missing_binary:
                    recommendations.append(f"Found {len(missing_binary)} sessions missing binary files.")
                if orphan_json:
                    recommendations.append(f"Found {len(orphan_json)} orphaned JSON files.")
                if orphan_binary:
                    recommendations.append(f"Found {len(orphan_binary)} orphaned binary files.")
                    
        except Exception as e:
            status = "failed"
            details['error'] = str(e)
            recommendations.append("Unable to check file consistency. Check file system permissions.")
            
        return IntegrityCheck(
            check_type="file_consistency",
            status=status,
            timestamp=time.time(),
            details=details,
            recommendations=recommendations
        )
        
    def _check_data_consistency(self) -> IntegrityCheck:
        """Check data consistency within SQLite database"""
        details = {}
        recommendations = []
        status = "passed"
        
        try:
            with self.db._get_sqlite_connection() as conn:
                # Check for orphaned token tests
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM token_tests tt
                    LEFT JOIN test_sessions ts ON tt.session_id = ts.session_id
                    WHERE ts.session_id IS NULL
                """)
                orphaned_tests = cursor.fetchone()[0]
                details['orphaned_token_tests'] = orphaned_tests
                
                if orphaned_tests > 0:
                    status = "warning"
                    recommendations.append(f"Found {orphaned_tests} token tests without corresponding sessions.")
                    
                # Check for missing token definitions
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM token_tests tt
                    LEFT JOIN token_definitions td ON tt.token_id = td.token_id
                    WHERE td.token_id IS NULL
                """)
                missing_definitions = cursor.fetchone()[0]
                details['missing_token_definitions'] = missing_definitions
                
                if missing_definitions > 0:
                    status = "warning"
                    recommendations.append(f"Found {missing_definitions} token tests referencing undefined tokens.")
                    
                # Check timestamp consistency
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM test_sessions ts
                    WHERE ts.end_timestamp < ts.start_timestamp
                """)
                invalid_timestamps = cursor.fetchone()[0]
                details['invalid_timestamps'] = invalid_timestamps
                
                if invalid_timestamps > 0:
                    status = "failed"
                    recommendations.append(f"Found {invalid_timestamps} sessions with invalid timestamps.")
                    
                # Check for duplicate test IDs
                cursor = conn.execute("""
                    SELECT COUNT(*) - COUNT(DISTINCT test_id) as duplicates
                    FROM token_tests
                """)
                duplicate_tests = cursor.fetchone()[0]
                details['duplicate_test_ids'] = duplicate_tests
                
                if duplicate_tests > 0:
                    status = "failed"
                    recommendations.append(f"Found {duplicate_tests} duplicate test IDs. This indicates data corruption.")
                    
        except Exception as e:
            status = "failed"
            details['error'] = str(e)
            recommendations.append("Unable to check data consistency. Database may be corrupted.")
            
        return IntegrityCheck(
            check_type="data_consistency",
            status=status,
            timestamp=time.time(),
            details=details,
            recommendations=recommendations
        )
        
    def _check_backup_integrity(self) -> IntegrityCheck:
        """Check backup system integrity"""
        details = {}
        recommendations = []
        status = "passed"
        
        try:
            backups = list(self.backup_dir.glob("backup_*"))
            details['backup_count'] = len(backups)
            
            if len(backups) == 0:
                status = "warning"
                recommendations.append("No backups found. Create at least one backup for data protection.")
            elif len(backups) < 3:
                status = "warning"
                recommendations.append("Less than 3 backups found. Consider maintaining more backup generations.")
                
            # Check backup ages
            current_time = time.time()
            backup_ages = []
            
            for backup_dir in backups:
                if backup_dir.is_dir():
                    manifest_file = backup_dir / "manifest.json"
                    if manifest_file.exists():
                        try:
                            with open(manifest_file, 'r') as f:
                                manifest = json.load(f)
                            backup_time = manifest.get('timestamp', 0)
                            age_hours = (current_time - backup_time) / 3600
                            backup_ages.append(age_hours)
                        except Exception as e:
                            logger.warning(f"Could not read backup manifest {manifest_file}: {str(e)}")
                            
            if backup_ages:
                details['newest_backup_age_hours'] = min(backup_ages)
                details['oldest_backup_age_hours'] = max(backup_ages)
                
                if min(backup_ages) > 24:  # Newest backup is over 24 hours old
                    status = "warning"
                    recommendations.append("Newest backup is over 24 hours old. Consider creating a fresh backup.")
                    
        except Exception as e:
            status = "failed"
            details['error'] = str(e)
            recommendations.append("Unable to check backup integrity. Check backup directory permissions.")
            
        return IntegrityCheck(
            check_type="backup_integrity",
            status=status,
            timestamp=time.time(),
            details=details,
            recommendations=recommendations
        )
        
    def _check_performance_health(self) -> IntegrityCheck:
        """Check database performance health"""
        details = {}
        recommendations = []
        status = "passed"
        
        try:
            with self.db._get_sqlite_connection() as conn:
                # Check database size
                cursor = conn.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor = conn.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                db_size_mb = (page_count * page_size) / (1024 * 1024)
                details['database_size_mb'] = db_size_mb
                
                # Check query performance with a simple test
                start_time = time.time()
                cursor = conn.execute("SELECT COUNT(*) FROM token_tests")
                test_count = cursor.fetchone()[0]
                query_time_ms = (time.time() - start_time) * 1000
                details['test_count'] = test_count
                details['count_query_time_ms'] = query_time_ms
                
                if query_time_ms > 1000:  # Over 1 second for count query
                    status = "warning"
                    recommendations.append("Database queries are slow. Consider rebuilding indices or running VACUUM.")
                    
                # Check for fragmentation
                cursor = conn.execute("PRAGMA freelist_count")
                freelist_count = cursor.fetchone()[0]
                details['freelist_count'] = freelist_count
                
                if freelist_count > page_count * 0.1:  # More than 10% fragmentation
                    status = "warning"
                    recommendations.append("Database fragmentation detected. Run VACUUM to optimize performance.")
                    
        except Exception as e:
            status = "failed"
            details['error'] = str(e)
            recommendations.append("Unable to check performance health. Database may be inaccessible.")
            
        return IntegrityCheck(
            check_type="performance_health",
            status=status,
            timestamp=time.time(),
            details=details,
            recommendations=recommendations
        )

class BackupManager:
    """Advanced backup management system"""
    
    def __init__(self, db_backend: DatabaseBackend):
        self.db = db_backend
        self.backup_dir = self.db.backup_dir
        
    def create_full_backup(self, backup_name: Optional[str] = None, 
                          compress: bool = True) -> BackupInfo:
        """Create a comprehensive backup with metadata"""
        if not backup_name:
            backup_name = f"full_backup_{int(time.time())}"
            
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        backup_info = BackupInfo(
            backup_id=f"bkp_{int(time.time())}",
            name=backup_name,
            timestamp=time.time(),
            size_bytes=0,
            file_count=0,
            checksum="",
            compression="gzip" if compress else "none",
            backup_path=str(backup_path),
            metadata={}
        )
        
        files_backed_up = 0
        total_size = 0
        
        # Backup SQLite database
        if self.db.sqlite_db.exists():
            dest = backup_path / "database.sqlite"
            shutil.copy2(self.db.sqlite_db, dest)
            files_backed_up += 1
            total_size += dest.stat().st_size
            
        # Backup JSON files
        if compress and self.db.json_dir.exists():
            json_archive = backup_path / "json_data.tar.gz"
            with tarfile.open(json_archive, 'w:gz') as tar:
                tar.add(self.db.json_dir, arcname='json')
            files_backed_up += 1
            total_size += json_archive.stat().st_size
        elif self.db.json_dir.exists():
            json_backup_dir = backup_path / "json"
            shutil.copytree(self.db.json_dir, json_backup_dir)
            files_backed_up += len(list(json_backup_dir.rglob("*")))
            total_size += sum(f.stat().st_size for f in json_backup_dir.rglob("*") if f.is_file())
            
        # Backup CSV files
        if compress and self.db.csv_dir.exists():
            csv_archive = backup_path / "csv_data.tar.gz"
            with tarfile.open(csv_archive, 'w:gz') as tar:
                tar.add(self.db.csv_dir, arcname='csv')
            files_backed_up += 1
            total_size += csv_archive.stat().st_size
        elif self.db.csv_dir.exists():
            csv_backup_dir = backup_path / "csv"
            shutil.copytree(self.db.csv_dir, csv_backup_dir)
            files_backed_up += len(list(csv_backup_dir.rglob("*")))
            total_size += sum(f.stat().st_size for f in csv_backup_dir.rglob("*") if f.is_file())
            
        # Backup binary files
        if compress and self.db.binary_dir.exists():
            binary_archive = backup_path / "binary_data.tar.gz"
            with tarfile.open(binary_archive, 'w:gz') as tar:
                tar.add(self.db.binary_dir, arcname='binary')
            files_backed_up += 1
            total_size += binary_archive.stat().st_size
        elif self.db.binary_dir.exists():
            binary_backup_dir = backup_path / "binary"
            shutil.copytree(self.db.binary_dir, binary_backup_dir)
            files_backed_up += len(list(binary_backup_dir.rglob("*")))
            total_size += sum(f.stat().st_size for f in binary_backup_dir.rglob("*") if f.is_file())
            
        # Calculate checksum
        backup_checksum = self._calculate_backup_checksum(backup_path)
        
        # Update backup info
        backup_info.size_bytes = total_size
        backup_info.file_count = files_backed_up
        backup_info.checksum = backup_checksum
        backup_info.metadata = {
            'created_by': 'IntegrityManager',
            'database_size': self.db.sqlite_db.stat().st_size if self.db.sqlite_db.exists() else 0,
            'compression_enabled': compress,
            'backup_type': 'full'
        }
        
        # Save backup manifest
        with open(backup_path / "manifest.json", 'w') as f:
            json.dump(asdict(backup_info), f, indent=2)
            
        logger.info(f"Created backup {backup_name}: {files_backed_up} files, {total_size / (1024*1024):.1f} MB")
        return backup_info
        
    def restore_from_backup(self, backup_name: str, verify_checksum: bool = True) -> bool:
        """Restore database from backup"""
        backup_path = self.backup_dir / backup_name
        manifest_file = backup_path / "manifest.json"
        
        if not backup_path.exists() or not manifest_file.exists():
            logger.error(f"Backup {backup_name} not found or incomplete")
            return False
            
        # Load backup manifest
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
            
        # Verify checksum if requested
        if verify_checksum:
            current_checksum = self._calculate_backup_checksum(backup_path)
            if current_checksum != manifest.get('checksum', ''):
                logger.error(f"Backup {backup_name} checksum verification failed")
                return False
                
        try:
            # Create backup of current state before restore
            emergency_backup = self.create_full_backup(f"pre_restore_{int(time.time())}")
            logger.info(f"Created emergency backup: {emergency_backup.name}")
            
            # Restore SQLite database
            sqlite_backup = backup_path / "database.sqlite"
            if sqlite_backup.exists():
                shutil.copy2(sqlite_backup, self.db.sqlite_db)
                logger.info("Restored SQLite database")
                
            # Restore JSON files
            json_archive = backup_path / "json_data.tar.gz"
            json_dir = backup_path / "json"
            
            if json_archive.exists():
                # Remove current JSON directory
                if self.db.json_dir.exists():
                    shutil.rmtree(self.db.json_dir)
                self.db.json_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract from archive
                with tarfile.open(json_archive, 'r:gz') as tar:
                    tar.extractall(self.db.json_dir.parent)
                logger.info("Restored JSON files from archive")
                
            elif json_dir.exists():
                # Remove current JSON directory
                if self.db.json_dir.exists():
                    shutil.rmtree(self.db.json_dir)
                shutil.copytree(json_dir, self.db.json_dir)
                logger.info("Restored JSON files from directory")
                
            # Similar restore process for CSV and binary files
            self._restore_data_files(backup_path, "csv", self.db.csv_dir)
            self._restore_data_files(backup_path, "binary", self.db.binary_dir)
            
            logger.info(f"Successfully restored from backup {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup {backup_name}: {str(e)}")
            return False
            
    def _restore_data_files(self, backup_path: Path, data_type: str, target_dir: Path):
        """Restore data files from backup"""
        archive_file = backup_path / f"{data_type}_data.tar.gz"
        backup_dir = backup_path / data_type
        
        if archive_file.exists():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            with tarfile.open(archive_file, 'r:gz') as tar:
                tar.extractall(target_dir.parent)
            logger.info(f"Restored {data_type} files from archive")
            
        elif backup_dir.exists():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(backup_dir, target_dir)
            logger.info(f"Restored {data_type} files from directory")
            
    def _calculate_backup_checksum(self, backup_path: Path) -> str:
        """Calculate checksum for backup verification"""
        hasher = hashlib.sha256()
        
        for file_path in sorted(backup_path.rglob("*")):
            if file_path.is_file() and file_path.name != "manifest.json":
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
                        
        return hasher.hexdigest()
        
    def list_backups(self) -> List[BackupInfo]:
        """List all available backups with metadata"""
        backups = []
        
        for backup_dir in self.backup_dir.glob("*"):
            if backup_dir.is_dir():
                manifest_file = backup_dir / "manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, 'r') as f:
                            manifest = json.load(f)
                        backup_info = BackupInfo(**manifest)
                        backups.append(backup_info)
                    except Exception as e:
                        logger.warning(f"Could not read backup manifest {manifest_file}: {str(e)}")
                        
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)
        
    def cleanup_old_backups(self, keep_count: int = 10, keep_days: int = 30):
        """Clean up old backups based on retention policy"""
        backups = self.list_backups()
        current_time = time.time()
        
        # Keep recent backups by count
        recent_backups = backups[:keep_count]
        
        # Keep backups within time window
        time_threshold = current_time - (keep_days * 24 * 3600)
        recent_by_time = [b for b in backups if b.timestamp > time_threshold]
        
        # Combine and deduplicate
        backups_to_keep = set(b.backup_id for b in recent_backups + recent_by_time)
        
        # Remove old backups
        removed_count = 0
        for backup in backups:
            if backup.backup_id not in backups_to_keep:
                backup_path = Path(backup.backup_path)
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                    removed_count += 1
                    logger.info(f"Removed old backup: {backup.name}")
                    
        logger.info(f"Cleaned up {removed_count} old backups, keeping {len(backups_to_keep)}")

if __name__ == "__main__":
    # Example usage
    from database_backend import DatabaseBackend
    
    db = DatabaseBackend()
    integrity_manager = IntegrityManager(db)
    backup_manager = BackupManager(db)
    
    # Run integrity checks
    checks = integrity_manager.run_integrity_checks()
    print(f"Ran {len(checks)} integrity checks")
    
    failed_checks = [c for c in checks if c.status == "failed"]
    warning_checks = [c for c in checks if c.status == "warning"]
    
    print(f"Failed checks: {len(failed_checks)}")
    print(f"Warning checks: {len(warning_checks)}")
    
    # Create backup
    backup_info = backup_manager.create_full_backup()
    print(f"Created backup: {backup_info.name} ({backup_info.size_bytes / (1024*1024):.1f} MB)")
    
    # List backups
    backups = backup_manager.list_backups()
    print(f"Total backups: {len(backups)}")
    
    # Example transaction usage
    with integrity_manager.transaction_manager.transaction() as txn:
        # Perform multiple operations atomically
        print(f"Transaction {txn['transaction_id']} started")
        # Operations would go here
        print("Transaction committed successfully")