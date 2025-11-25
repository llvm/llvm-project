#!/usr/bin/env python3
"""
DSMIL Database Management Console
Main interface for database operations, monitoring, and maintenance
Version: 1.0.0
Date: 2025-09-01
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add backend modules to path
sys.path.insert(0, str(Path(__file__).parent / 'backends'))
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))
sys.path.insert(0, str(Path(__file__).parent / 'analysis'))
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from database_backend import DatabaseBackend
from auto_recorder import AutoRecorder, RecordingSession
from query_analyzer import QueryAnalyzer
from integrity_manager import IntegrityManager, BackupManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Main database management interface"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.base_path = Path(__file__).parent
        self.config = self._load_config(config_path)
        
        # Initialize backend systems
        self.db = DatabaseBackend(str(self.base_path))
        self.recorder = AutoRecorder(str(self.base_path))
        self.analyzer = QueryAnalyzer(self.db)
        self.integrity_manager = IntegrityManager(self.db)
        self.backup_manager = BackupManager(self.db)
        
        logger.info("Database management system initialized")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file"""
        if not config_path:
            config_path = self.base_path / "config" / "database_config.json"
            
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}, using defaults")
            return {}
            
        with open(config_file, 'r') as f:
            return json.load(f)
            
    def status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'timestamp': time.time(),
            'database': {},
            'integrity': {},
            'backups': {},
            'monitoring': {}
        }
        
        # Database status
        try:
            with self.db._get_sqlite_connection() as conn:
                # Get basic counts
                cursor = conn.execute("SELECT COUNT(*) FROM test_sessions")
                session_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM token_tests")
                test_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM thermal_readings")
                thermal_count = cursor.fetchone()[0]
                
                # Get database size
                db_size = self.db.sqlite_db.stat().st_size if self.db.sqlite_db.exists() else 0
                
                status['database'] = {
                    'available': True,
                    'size_mb': db_size / (1024 * 1024),
                    'sessions': session_count,
                    'tests': test_count,
                    'thermal_readings': thermal_count
                }
                
        except Exception as e:
            status['database'] = {'available': False, 'error': str(e)}
            
        # Integrity status
        try:
            integrity_result = self.db.verify_data_integrity()
            status['integrity'] = integrity_result
        except Exception as e:
            status['integrity'] = {'error': str(e)}
            
        # Backup status
        try:
            backups = self.backup_manager.list_backups()
            status['backups'] = {
                'count': len(backups),
                'latest': backups[0].timestamp if backups else None,
                'total_size_mb': sum(b.size_bytes for b in backups) / (1024 * 1024)
            }
        except Exception as e:
            status['backups'] = {'error': str(e)}
            
        # Monitoring status
        status['monitoring'] = {
            'recorder_active': self.recorder.running,
            'current_session': self.recorder.current_session
        }
        
        return status
        
    def initialize(self) -> bool:
        """Initialize database with schema and sample data"""
        try:
            logger.info("Initializing database schema...")
            self.db._initialize_sqlite()
            
            logger.info("Initializing CSV files...")
            self.db._initialize_csv_files()
            
            logger.info("Database initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            return False
            
    def create_backup(self, name: Optional[str] = None) -> bool:
        """Create database backup"""
        try:
            backup_info = self.backup_manager.create_full_backup(name)
            logger.info(f"Backup created: {backup_info.name} ({backup_info.size_bytes / (1024*1024):.1f} MB)")
            return True
        except Exception as e:
            logger.error(f"Backup creation failed: {str(e)}")
            return False
            
    def restore_backup(self, name: str) -> bool:
        """Restore from backup"""
        try:
            success = self.backup_manager.restore_from_backup(name)
            if success:
                logger.info(f"Successfully restored from backup: {name}")
            else:
                logger.error(f"Failed to restore from backup: {name}")
            return success
        except Exception as e:
            logger.error(f"Backup restoration failed: {str(e)}")
            return False
            
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        try:
            backups = self.backup_manager.list_backups()
            return [
                {
                    'name': b.name,
                    'timestamp': b.timestamp,
                    'size_mb': b.size_bytes / (1024 * 1024),
                    'file_count': b.file_count,
                    'age_hours': (time.time() - b.timestamp) / 3600
                }
                for b in backups
            ]
        except Exception as e:
            logger.error(f"Failed to list backups: {str(e)}")
            return []
            
    def run_integrity_checks(self) -> bool:
        """Run comprehensive integrity checks"""
        try:
            checks = self.integrity_manager.run_integrity_checks()
            
            passed = len([c for c in checks if c.status == "passed"])
            warning = len([c for c in checks if c.status == "warning"])
            failed = len([c for c in checks if c.status == "failed"])
            
            logger.info(f"Integrity checks completed: {passed} passed, {warning} warnings, {failed} failed")
            
            # Print details for failed and warning checks
            for check in checks:
                if check.status in ["failed", "warning"]:
                    logger.warning(f"{check.check_type}: {check.status}")
                    for recommendation in check.recommendations:
                        logger.warning(f"  - {recommendation}")
                        
            return failed == 0
            
        except Exception as e:
            logger.error(f"Integrity checks failed: {str(e)}")
            return False
            
    def start_recording(self, session_name: str, session_type: str = "manual", 
                       operator: Optional[str] = None) -> str:
        """Start recording session"""
        try:
            session_id = self.recorder.start_session(session_name, session_type, operator)
            logger.info(f"Started recording session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to start recording: {str(e)}")
            return ""
            
    def stop_recording(self, status: str = "completed", notes: Optional[str] = None):
        """Stop current recording session"""
        try:
            if self.recorder.current_session:
                self.recorder.stop_session(status, notes)
                logger.info("Recording session stopped")
            else:
                logger.warning("No active recording session")
        except Exception as e:
            logger.error(f"Failed to stop recording: {str(e)}")
            
    def analyze_session(self, session_id: str) -> bool:
        """Analyze specific session"""
        try:
            analysis = self.analyzer.analyze_session(session_id)
            
            # Print summary
            print(f"\nSession Analysis: {session_id}")
            print("=" * 50)
            print(f"Total tests: {analysis.summary['total_tests']}")
            print(f"Success rate: {analysis.summary['success_rate']:.1%}")
            print(f"Duration: {analysis.summary['duration_hours']:.1f} hours")
            print(f"Thermal events: {analysis.summary['thermal_events']}")
            print(f"Patterns found: {analysis.summary['pattern_count']}")
            
            # Print recommendations
            if analysis.recommendations:
                print("\nRecommendations:")
                for i, rec in enumerate(analysis.recommendations, 1):
                    print(f"{i}. {rec}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Session analysis failed: {str(e)}")
            return False
            
    def analyze_tokens(self, token_id: Optional[int] = None) -> bool:
        """Analyze token performance"""
        try:
            analysis = self.analyzer.analyze_token_performance(token_id)
            
            # Print summary
            print(f"\nToken Performance Analysis")
            print("=" * 50)
            print(f"Tokens analyzed: {analysis.summary['tokens_analyzed']}")
            print(f"Total tests: {analysis.summary['total_tests']}")
            print(f"Overall success rate: {analysis.summary['overall_success_rate']:.1%}")
            print(f"Average duration: {analysis.summary['avg_test_duration']:.1f} ms")
            
            # Show top performing tokens
            token_data = analysis.details['token_performance']
            top_tokens = sorted(token_data, key=lambda x: x.get('successful_tests', 0), reverse=True)[:10]
            
            print("\nTop Performing Tokens:")
            print("Token ID | Hex ID | Success Rate | Tests")
            print("-" * 40)
            for token in top_tokens:
                success_rate = (token['successful_tests'] / token['total_tests']) if token['total_tests'] > 0 else 0
                print(f"{token['token_id']:8} | {token['hex_id']:6} | {success_rate:10.1%} | {token['total_tests']:5}")
                
            return True
            
        except Exception as e:
            logger.error(f"Token analysis failed: {str(e)}")
            return False
            
    def generate_report(self, session_ids: Optional[List[str]] = None, output_file: Optional[str] = None) -> bool:
        """Generate comprehensive analysis report"""
        try:
            report = self.analyzer.generate_comprehensive_report(session_ids)
            
            if output_file:
                output_path = Path(output_file)
            else:
                output_path = self.base_path / "analysis" / f"report_{int(time.time())}.json"
                
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            logger.info(f"Report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return False
            
    def cleanup(self, keep_backups: int = 10, keep_days: int = 30):
        """Clean up old data and backups"""
        try:
            # Clean up old backups
            self.backup_manager.cleanup_old_backups(keep_backups, keep_days)
            
            # Run database maintenance
            with self.db._get_sqlite_connection() as conn:
                logger.info("Running database VACUUM...")
                conn.execute("VACUUM")
                logger.info("Running database ANALYZE...")
                conn.execute("ANALYZE")
                
            logger.info("Cleanup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return False

def main():
    """Command-line interface for database management"""
    parser = argparse.ArgumentParser(description="DSMIL Database Management Console")
    parser.add_argument('--config', '-c', help="Configuration file path")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show system status')
    
    # Initialize command
    subparsers.add_parser('init', help='Initialize database')
    
    # Backup commands
    backup_parser = subparsers.add_parser('backup', help='Backup operations')
    backup_parser.add_argument('action', choices=['create', 'restore', 'list'], help='Backup action')
    backup_parser.add_argument('--name', '-n', help='Backup name')
    
    # Integrity commands
    subparsers.add_parser('check', help='Run integrity checks')
    
    # Recording commands
    record_parser = subparsers.add_parser('record', help='Recording operations')
    record_parser.add_argument('action', choices=['start', 'stop'], help='Recording action')
    record_parser.add_argument('--name', '-n', help='Session name')
    record_parser.add_argument('--type', '-t', default='manual', help='Session type')
    record_parser.add_argument('--operator', '-o', help='Operator name')
    record_parser.add_argument('--status', '-s', default='completed', help='Stop status')
    record_parser.add_argument('--notes', help='Session notes')
    
    # Analysis commands
    analyze_parser = subparsers.add_parser('analyze', help='Analysis operations')
    analyze_parser.add_argument('type', choices=['session', 'tokens', 'thermal'], help='Analysis type')
    analyze_parser.add_argument('--session', '-s', help='Session ID for analysis')
    analyze_parser.add_argument('--token', '-t', type=int, help='Token ID for analysis')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate comprehensive report')
    report_parser.add_argument('--sessions', '-s', nargs='+', help='Session IDs to include')
    report_parser.add_argument('--output', '-o', help='Output file path')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument('--keep-backups', type=int, default=10, help='Number of backups to keep')
    cleanup_parser.add_argument('--keep-days', type=int, default=30, help='Days to keep backups')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    # Initialize database manager
    manager = DatabaseManager(args.config)
    
    # Execute commands
    try:
        if args.command == 'status':
            status = manager.status()
            print(json.dumps(status, indent=2, default=str))
            
        elif args.command == 'init':
            if manager.initialize():
                print("Database initialized successfully")
            else:
                print("Database initialization failed")
                sys.exit(1)
                
        elif args.command == 'backup':
            if args.action == 'create':
                if manager.create_backup(args.name):
                    print("Backup created successfully")
                else:
                    print("Backup creation failed")
                    sys.exit(1)
                    
            elif args.action == 'restore':
                if not args.name:
                    print("Backup name required for restore")
                    sys.exit(1)
                if manager.restore_backup(args.name):
                    print("Backup restored successfully")
                else:
                    print("Backup restoration failed")
                    sys.exit(1)
                    
            elif args.action == 'list':
                backups = manager.list_backups()
                if backups:
                    print("Available Backups:")
                    print("Name                    | Age (hours) | Size (MB) | Files")
                    print("-" * 60)
                    for backup in backups:
                        print(f"{backup['name']:22} | {backup['age_hours']:9.1f} | {backup['size_mb']:7.1f} | {backup['file_count']:5}")
                else:
                    print("No backups found")
                    
        elif args.command == 'check':
            if manager.run_integrity_checks():
                print("All integrity checks passed")
            else:
                print("Some integrity checks failed")
                sys.exit(1)
                
        elif args.command == 'record':
            if args.action == 'start':
                if not args.name:
                    print("Session name required")
                    sys.exit(1)
                session_id = manager.start_recording(args.name, args.type, args.operator)
                if session_id:
                    print(f"Recording started: {session_id}")
                else:
                    print("Failed to start recording")
                    sys.exit(1)
                    
            elif args.action == 'stop':
                manager.stop_recording(args.status, args.notes)
                print("Recording stopped")
                
        elif args.command == 'analyze':
            if args.type == 'session':
                if not args.session:
                    print("Session ID required for session analysis")
                    sys.exit(1)
                if not manager.analyze_session(args.session):
                    sys.exit(1)
                    
            elif args.type == 'tokens':
                if not manager.analyze_tokens(args.token):
                    sys.exit(1)
                    
            elif args.type == 'thermal':
                analysis = manager.analyzer.detect_thermal_correlations()
                print("Thermal correlation analysis completed")
                print(f"Found {analysis.summary['correlations_found']} correlations")
                
        elif args.command == 'report':
            if manager.generate_report(args.sessions, args.output):
                print("Report generated successfully")
            else:
                print("Report generation failed")
                sys.exit(1)
                
        elif args.command == 'cleanup':
            if manager.cleanup(args.keep_backups, args.keep_days):
                print("Cleanup completed successfully")
            else:
                print("Cleanup failed")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()