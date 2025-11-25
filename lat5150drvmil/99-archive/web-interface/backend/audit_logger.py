#!/usr/bin/env python3
"""
DSMIL Audit Logger
Comprehensive audit logging for all system activities
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Audit log entry"""
    entry_id: str
    sequence_number: int
    timestamp: datetime
    event_type: str
    user_id: str
    session_id: Optional[str] = None
    device_id: Optional[int] = None
    operation_type: Optional[str] = None
    risk_level: Optional[str] = None
    authorization_result: str = "UNKNOWN"
    operation_result: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    system_context: Optional[Dict[str, Any]] = None
    integrity_hash: Optional[str] = None
    chain_hash: Optional[str] = None


class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self):
        self.sequence_counter = 0
        self.last_entry_hash = None
        self.audit_entries: List[AuditEntry] = []
        self.max_memory_entries = 10000  # Keep last 10k entries in memory
        
    async def log_event(
        self,
        event_type: str,
        user_id: str,
        session_id: Optional[str] = None,
        device_id: Optional[int] = None,
        operation_type: Optional[str] = None,
        risk_level: Optional[str] = None,
        authorization_result: str = "AUTHORIZED",
        operation_result: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        system_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Log audit event with integrity protection"""
        
        try:
            # Generate unique entry ID
            entry_id = str(uuid.uuid4())
            
            # Increment sequence number
            self.sequence_counter += 1
            
            # Create audit entry
            entry = AuditEntry(
                entry_id=entry_id,
                sequence_number=self.sequence_counter,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                device_id=device_id,
                operation_type=operation_type,
                risk_level=risk_level,
                authorization_result=authorization_result,
                operation_result=operation_result,
                details=details or {},
                system_context=system_context or self._get_system_context()
            )
            
            # Generate integrity hash
            entry.integrity_hash = self._generate_integrity_hash(entry)
            entry.chain_hash = self._generate_chain_hash(entry)
            
            # Store entry
            self.audit_entries.append(entry)
            
            # Maintain memory limit
            if len(self.audit_entries) > self.max_memory_entries:
                self.audit_entries = self.audit_entries[-self.max_memory_entries:]
            
            # Update last entry hash for chaining
            self.last_entry_hash = entry.integrity_hash
            
            # Log to standard logger as well
            log_level = self._get_log_level(event_type, risk_level)
            logger.log(
                log_level,
                f"AUDIT [{event_type}] User: {user_id}, Device: {device_id}, "
                f"Operation: {operation_type}, Result: {operation_result}, "
                f"Risk: {risk_level}, Auth: {authorization_result}"
            )
            
            # In production, this would also write to database
            await self._persist_to_database(entry)
            
            return {
                "entry_id": entry_id,
                "sequence_number": self.sequence_counter,
                "timestamp": entry.timestamp.isoformat(),
                "integrity_hash": entry.integrity_hash
            }
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            # Still try to create a minimal audit record
            return {
                "entry_id": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _generate_integrity_hash(self, entry: AuditEntry) -> str:
        """Generate integrity hash for audit entry"""
        # Create deterministic string representation
        hash_data = {
            "entry_id": entry.entry_id,
            "sequence_number": entry.sequence_number,
            "timestamp": entry.timestamp.isoformat(),
            "event_type": entry.event_type,
            "user_id": entry.user_id,
            "session_id": entry.session_id,
            "device_id": entry.device_id,
            "operation_type": entry.operation_type,
            "risk_level": entry.risk_level,
            "authorization_result": entry.authorization_result,
            "operation_result": entry.operation_result,
            "details": entry.details,
            "system_context": entry.system_context
        }
        
        # Create JSON string with sorted keys for consistency
        hash_string = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()
    
    def _generate_chain_hash(self, entry: AuditEntry) -> str:
        """Generate chain hash linking to previous entry"""
        if not self.last_entry_hash:
            return "0" * 64  # Genesis entry
        
        chain_data = f"{self.last_entry_hash}{entry.integrity_hash}"
        return hashlib.sha256(chain_data.encode('utf-8')).hexdigest()
    
    def _get_system_context(self) -> Dict[str, Any]:
        """Get current system context"""
        return {
            "system_time": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": "production" if not settings.debug_mode else "development"
        }
    
    def _get_log_level(self, event_type: str, risk_level: Optional[str] = None) -> int:
        """Determine log level based on event type and risk"""
        if event_type in ["EMERGENCY_STOP_TRIGGERED", "SECURITY_BREACH", "UNAUTHORIZED_ACCESS"]:
            return logging.CRITICAL
        elif event_type in ["OPERATION_DENIED", "AUTHENTICATION_FAILURE", "HIGH_RISK_OPERATION"]:
            return logging.ERROR
        elif event_type in ["DEVICE_OPERATION_FAILED", "UNUSUAL_ACTIVITY"]:
            return logging.WARNING
        elif risk_level in ["HIGH", "CRITICAL"]:
            return logging.WARNING
        else:
            return logging.INFO
    
    async def _persist_to_database(self, entry: AuditEntry):
        """Persist audit entry to database"""
        # This would write to PostgreSQL in production
        # For now, we'll just simulate the database operation
        try:
            # Simulate database write
            pass
        except Exception as e:
            logger.error(f"Failed to persist audit entry to database: {e}")
    
    async def query_audit_log(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        device_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query audit log entries"""
        
        # Filter entries based on criteria
        filtered_entries = self.audit_entries.copy()
        
        if user_id:
            filtered_entries = [e for e in filtered_entries if e.user_id == user_id]
        
        if event_type:
            filtered_entries = [e for e in filtered_entries if e.event_type == event_type]
        
        if device_id:
            filtered_entries = [e for e in filtered_entries if e.device_id == device_id]
        
        if start_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]
        
        if end_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_time]
        
        # Sort by timestamp (newest first) and limit
        filtered_entries.sort(key=lambda e: e.timestamp, reverse=True)
        filtered_entries = filtered_entries[:limit]
        
        # Convert to dictionaries
        return [asdict(entry) for entry in filtered_entries]
    
    async def verify_audit_chain_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit log chain"""
        try:
            total_entries = len(self.audit_entries)
            verified_entries = 0
            broken_chains = []
            invalid_hashes = []
            
            previous_hash = None
            
            for i, entry in enumerate(self.audit_entries):
                # Verify integrity hash
                calculated_hash = self._generate_integrity_hash(entry)
                if calculated_hash != entry.integrity_hash:
                    invalid_hashes.append({
                        "sequence": entry.sequence_number,
                        "entry_id": entry.entry_id,
                        "expected": calculated_hash,
                        "actual": entry.integrity_hash
                    })
                else:
                    verified_entries += 1
                
                # Verify chain hash (skip first entry)
                if i > 0:
                    expected_chain_hash = hashlib.sha256(
                        f"{previous_hash}{entry.integrity_hash}".encode('utf-8')
                    ).hexdigest()
                    
                    if expected_chain_hash != entry.chain_hash:
                        broken_chains.append({
                            "sequence": entry.sequence_number,
                            "entry_id": entry.entry_id,
                            "expected": expected_chain_hash,
                            "actual": entry.chain_hash
                        })
                
                previous_hash = entry.integrity_hash
            
            integrity_percentage = (verified_entries / total_entries * 100) if total_entries > 0 else 100
            
            return {
                "total_entries": total_entries,
                "verified_entries": verified_entries,
                "integrity_percentage": integrity_percentage,
                "invalid_hashes": invalid_hashes,
                "broken_chains": broken_chains,
                "is_valid": len(invalid_hashes) == 0 and len(broken_chains) == 0,
                "verification_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to verify audit chain integrity: {e}")
            return {
                "error": str(e),
                "is_valid": False,
                "verification_time": datetime.utcnow().isoformat()
            }
    
    async def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics"""
        try:
            if not self.audit_entries:
                return {
                    "total_entries": 0,
                    "oldest_entry": None,
                    "newest_entry": None,
                    "event_types": {},
                    "users": {},
                    "devices": {},
                    "risk_levels": {}
                }
            
            # Count by event type
            event_types = {}
            users = {}
            devices = {}
            risk_levels = {}
            
            for entry in self.audit_entries:
                # Event types
                event_types[entry.event_type] = event_types.get(entry.event_type, 0) + 1
                
                # Users
                users[entry.user_id] = users.get(entry.user_id, 0) + 1
                
                # Devices
                if entry.device_id:
                    devices[str(entry.device_id)] = devices.get(str(entry.device_id), 0) + 1
                
                # Risk levels
                if entry.risk_level:
                    risk_levels[entry.risk_level] = risk_levels.get(entry.risk_level, 0) + 1
            
            return {
                "total_entries": len(self.audit_entries),
                "oldest_entry": self.audit_entries[0].timestamp.isoformat(),
                "newest_entry": self.audit_entries[-1].timestamp.isoformat(),
                "event_types": event_types,
                "users": users,
                "devices": devices,
                "risk_levels": risk_levels,
                "current_sequence": self.sequence_counter
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {"error": str(e)}
    
    async def export_audit_log(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export audit log for external analysis"""
        try:
            # Filter entries by time range
            filtered_entries = self.audit_entries.copy()
            
            if start_time:
                filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]
            
            if end_time:
                filtered_entries = [e for e in filtered_entries if e.timestamp <= end_time]
            
            # Convert to export format
            if format.lower() == "json":
                export_data = {
                    "export_time": datetime.utcnow().isoformat(),
                    "entry_count": len(filtered_entries),
                    "start_time": start_time.isoformat() if start_time else None,
                    "end_time": end_time.isoformat() if end_time else None,
                    "entries": [asdict(entry) for entry in filtered_entries]
                }
                
                return {
                    "format": "json",
                    "data": export_data,
                    "size_bytes": len(json.dumps(export_data))
                }
            
            else:
                return {"error": f"Unsupported export format: {format}"}
                
        except Exception as e:
            logger.error(f"Failed to export audit log: {e}")
            return {"error": str(e)}


# Global audit logger instance
audit_logger = AuditLogger()