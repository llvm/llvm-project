#!/usr/bin/env python3
"""
DSMIL WebSocket Manager
Real-time communication for device updates and system monitoring
"""

import json
import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from fastapi import WebSocket

from auth import UserContext

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConnection:
    """WebSocket connection information"""
    connection_id: str
    websocket: WebSocket
    user_context: UserContext
    connected_at: datetime
    last_ping: datetime
    subscriptions: Set[str]


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: str
    data: Any
    timestamp: datetime
    source: Optional[str] = None
    target: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "target": self.target
        }


class WebSocketManager:
    """WebSocket connection and message management"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        self.subscription_groups: Dict[str, Set[str]] = {}  # subscription -> connection_ids
        self.message_queue: Dict[str, List[WebSocketMessage]] = {}  # connection_id -> messages
        
        # Message types
        self.MESSAGE_TYPES = {
            "SYSTEM_STATUS_UPDATE",
            "DEVICE_STATE_CHANGED", 
            "SECURITY_ALERT",
            "AUDIT_EVENT",
            "OPERATION_RESULT",
            "EMERGENCY_STOP",
            "USER_MESSAGE",
            "SYSTEM_NOTIFICATION",
            "PERFORMANCE_METRICS",
            "CONNECTION_STATUS",
            "PING",
            "PONG"
        }
        
        # Start background tasks
        asyncio.create_task(self._ping_connections_loop())
        asyncio.create_task(self._cleanup_connections_loop())
    
    async def add_connection(self, websocket: WebSocket, user_context: UserContext) -> str:
        """Add new WebSocket connection"""
        connection_id = str(uuid.uuid4())
        
        connection = WebSocketConnection(
            connection_id=connection_id,
            websocket=websocket,
            user_context=user_context,
            connected_at=datetime.utcnow(),
            last_ping=datetime.utcnow(),
            subscriptions={"system_status", "security_alerts"}  # Default subscriptions
        )
        
        self.connections[connection_id] = connection
        
        # Track user connections
        if user_context.user_id not in self.user_connections:
            self.user_connections[user_context.user_id] = set()
        self.user_connections[user_context.user_id].add(connection_id)
        
        # Add to subscription groups
        for subscription in connection.subscriptions:
            if subscription not in self.subscription_groups:
                self.subscription_groups[subscription] = set()
            self.subscription_groups[subscription].add(connection_id)
        
        # Send welcome message
        await self.send_message_to_connection(
            connection_id,
            WebSocketMessage(
                type="CONNECTION_STATUS",
                data={
                    "status": "connected",
                    "connection_id": connection_id,
                    "user_id": user_context.user_id,
                    "subscriptions": list(connection.subscriptions),
                    "server_time": datetime.utcnow().isoformat()
                },
                timestamp=datetime.utcnow(),
                source="server"
            )
        )
        
        logger.info(f"WebSocket connection {connection_id} established for user {user_context.username}")
        return connection_id
    
    async def remove_connection(self, connection_id: str):
        """Remove WebSocket connection"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        user_id = connection.user_context.user_id
        
        # Remove from user connections
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Remove from subscription groups
        for subscription in connection.subscriptions:
            if subscription in self.subscription_groups:
                self.subscription_groups[subscription].discard(connection_id)
                if not self.subscription_groups[subscription]:
                    del self.subscription_groups[subscription]
        
        # Remove from connections
        del self.connections[connection_id]
        
        # Clear message queue
        if connection_id in self.message_queue:
            del self.message_queue[connection_id]
        
        logger.info(f"WebSocket connection {connection_id} removed for user {connection.user_context.username}")
    
    async def send_message_to_connection(self, connection_id: str, message: WebSocketMessage):
        """Send message to specific connection"""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        try:
            await connection.websocket.send_text(json.dumps(message.to_dict()))
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to connection {connection_id}: {e}")
            # Queue message for retry if connection is still valid
            if connection_id in self.connections:
                if connection_id not in self.message_queue:
                    self.message_queue[connection_id] = []
                self.message_queue[connection_id].append(message)
            return False
    
    async def send_message_to_user(self, user_id: str, message: WebSocketMessage):
        """Send message to all connections of a user"""
        if user_id not in self.user_connections:
            return 0
        
        sent_count = 0
        for connection_id in self.user_connections[user_id].copy():
            if await self.send_message_to_connection(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_subscription(self, subscription: str, message: WebSocketMessage):
        """Broadcast message to all connections subscribed to a topic"""
        if subscription not in self.subscription_groups:
            return 0
        
        sent_count = 0
        for connection_id in self.subscription_groups[subscription].copy():
            if await self.send_message_to_connection(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_all(self, message: WebSocketMessage):
        """Broadcast message to all connected clients"""
        sent_count = 0
        for connection_id in list(self.connections.keys()):
            if await self.send_message_to_connection(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def handle_message(self, connection_id: str, data: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        message_type = data.get("type")
        message_data = data.get("data", {})
        
        try:
            if message_type == "PING":
                # Respond with PONG
                await self.send_message_to_connection(
                    connection_id,
                    WebSocketMessage(
                        type="PONG",
                        data={"timestamp": datetime.utcnow().isoformat()},
                        timestamp=datetime.utcnow(),
                        source="server"
                    )
                )
                connection.last_ping = datetime.utcnow()
                
            elif message_type == "SUBSCRIBE":
                # Handle subscription request
                subscription = message_data.get("subscription")
                if subscription and self._is_valid_subscription(subscription, connection.user_context):
                    await self._add_subscription(connection_id, subscription)
                    
            elif message_type == "UNSUBSCRIBE":
                # Handle unsubscription request
                subscription = message_data.get("subscription")
                if subscription:
                    await self._remove_subscription(connection_id, subscription)
                    
            elif message_type == "USER_MESSAGE":
                # Handle user-to-user messages (if authorized)
                target_user = message_data.get("target_user")
                if target_user and self._can_send_user_message(connection.user_context, target_user):
                    await self.send_message_to_user(
                        target_user,
                        WebSocketMessage(
                            type="USER_MESSAGE",
                            data=message_data,
                            timestamp=datetime.utcnow(),
                            source=connection.user_context.user_id,
                            target=target_user
                        )
                    )
                    
            else:
                logger.warning(f"Unknown message type '{message_type}' from connection {connection_id}")
                
        except Exception as e:
            logger.error(f"Error handling message from connection {connection_id}: {e}")
    
    def _is_valid_subscription(self, subscription: str, user_context: UserContext) -> bool:
        """Check if user can subscribe to topic"""
        # Define subscription access rules
        subscription_access = {
            "system_status": ["RESTRICTED", "CONFIDENTIAL", "SECRET", "TOP_SECRET"],
            "security_alerts": ["SECRET", "TOP_SECRET"],
            "audit_events": ["SECRET", "TOP_SECRET"], 
            "device_updates": ["CONFIDENTIAL", "SECRET", "TOP_SECRET"],
            "performance_metrics": ["RESTRICTED", "CONFIDENTIAL", "SECRET", "TOP_SECRET"],
            "emergency_notifications": ["RESTRICTED", "CONFIDENTIAL", "SECRET", "TOP_SECRET"]
        }
        
        required_clearances = subscription_access.get(subscription, ["TOP_SECRET"])
        return user_context.clearance_level in required_clearances
    
    def _can_send_user_message(self, sender: UserContext, target_user_id: str) -> bool:
        """Check if user can send message to another user"""
        # For now, allow messages between users of same or lower clearance
        # In production, implement more sophisticated access control
        return True
    
    async def _add_subscription(self, connection_id: str, subscription: str):
        """Add subscription for connection"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            connection.subscriptions.add(subscription)
            
            if subscription not in self.subscription_groups:
                self.subscription_groups[subscription] = set()
            self.subscription_groups[subscription].add(connection_id)
            
            # Confirm subscription
            await self.send_message_to_connection(
                connection_id,
                WebSocketMessage(
                    type="SUBSCRIPTION_CONFIRMED",
                    data={"subscription": subscription},
                    timestamp=datetime.utcnow(),
                    source="server"
                )
            )
            
            logger.info(f"Connection {connection_id} subscribed to {subscription}")
    
    async def _remove_subscription(self, connection_id: str, subscription: str):
        """Remove subscription for connection"""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            connection.subscriptions.discard(subscription)
            
            if subscription in self.subscription_groups:
                self.subscription_groups[subscription].discard(connection_id)
            
            # Confirm unsubscription
            await self.send_message_to_connection(
                connection_id,
                WebSocketMessage(
                    type="UNSUBSCRIPTION_CONFIRMED", 
                    data={"subscription": subscription},
                    timestamp=datetime.utcnow(),
                    source="server"
                )
            )
            
            logger.info(f"Connection {connection_id} unsubscribed from {subscription}")
    
    async def _ping_connections_loop(self):
        """Background task to ping connections and check connectivity"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for connection_id, connection in list(self.connections.items()):
                    # Check if connection is stale
                    if (current_time - connection.last_ping).total_seconds() > 60:  # 1 minute timeout
                        try:
                            await self.send_message_to_connection(
                                connection_id,
                                WebSocketMessage(
                                    type="PING",
                                    data={"timestamp": current_time.isoformat()},
                                    timestamp=current_time,
                                    source="server"
                                )
                            )
                        except Exception as e:
                            logger.warning(f"Connection {connection_id} appears dead: {e}")
                            await self.remove_connection(connection_id)
                
                await asyncio.sleep(30)  # Ping every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ping loop error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_connections_loop(self):
        """Background task to cleanup dead connections"""
        while True:
            try:
                current_time = datetime.utcnow()
                dead_connections = []
                
                for connection_id, connection in self.connections.items():
                    # Remove connections that haven't pinged in 5 minutes
                    if (current_time - connection.last_ping).total_seconds() > 300:
                        dead_connections.append(connection_id)
                
                for connection_id in dead_connections:
                    logger.info(f"Cleaning up dead connection {connection_id}")
                    await self.remove_connection(connection_id)
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)
    
    # High-level messaging methods for system components
    
    async def broadcast_system_status(self, status_data: Dict[str, Any]):
        """Broadcast system status update"""
        message = WebSocketMessage(
            type="SYSTEM_STATUS_UPDATE",
            data=status_data,
            timestamp=datetime.utcnow(),
            source="system_monitor"
        )
        
        return await self.broadcast_to_subscription("system_status", message)
    
    async def broadcast_security_alert(self, alert_data: Dict[str, Any]):
        """Broadcast security alert"""
        message = WebSocketMessage(
            type="SECURITY_ALERT",
            data=alert_data,
            timestamp=datetime.utcnow(),
            source="security_monitor"
        )
        
        return await self.broadcast_to_subscription("security_alerts", message)
    
    async def broadcast_device_update(self, device_id: int, update_data: Dict[str, Any]):
        """Broadcast device status update"""
        message = WebSocketMessage(
            type="DEVICE_STATE_CHANGED",
            data={
                "device_id": device_id,
                "update": update_data
            },
            timestamp=datetime.utcnow(),
            source="device_controller"
        )
        
        return await self.broadcast_to_subscription("device_updates", message)
    
    async def broadcast_audit_event(self, audit_data: Dict[str, Any]):
        """Broadcast audit event"""
        message = WebSocketMessage(
            type="AUDIT_EVENT", 
            data=audit_data,
            timestamp=datetime.utcnow(),
            source="audit_logger"
        )
        
        return await self.broadcast_to_subscription("audit_events", message)
    
    async def broadcast_emergency_stop(self, reason: str):
        """Broadcast emergency stop activation"""
        message = WebSocketMessage(
            type="EMERGENCY_STOP",
            data={
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "CRITICAL"
            },
            timestamp=datetime.utcnow(),
            source="emergency_system"
        )
        
        return await self.broadcast_to_all(message)
    
    async def send_operation_result(self, user_id: str, operation_data: Dict[str, Any]):
        """Send operation result to specific user"""
        message = WebSocketMessage(
            type="OPERATION_RESULT",
            data=operation_data,
            timestamp=datetime.utcnow(),
            source="device_controller",
            target=user_id
        )
        
        return await self.send_message_to_user(user_id, message)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        user_count = len(self.user_connections)
        total_connections = len(self.connections)
        subscription_stats = {
            subscription: len(connections)
            for subscription, connections in self.subscription_groups.items()
        }
        
        return {
            "total_connections": total_connections,
            "unique_users": user_count,
            "subscriptions": subscription_stats,
            "message_queue_size": sum(len(queue) for queue in self.message_queue.values())
        }
    
    async def cleanup(self):
        """Cleanup WebSocket manager"""
        logger.info("Cleaning up WebSocket manager...")
        
        # Close all connections
        for connection_id in list(self.connections.keys()):
            try:
                connection = self.connections[connection_id]
                await connection.websocket.close()
            except Exception as e:
                logger.error(f"Error closing connection {connection_id}: {e}")
        
        # Clear all data structures
        self.connections.clear()
        self.user_connections.clear()
        self.subscription_groups.clear()
        self.message_queue.clear()
        
        logger.info("WebSocket manager cleanup complete")


# Global WebSocket manager instance
websocket_manager = WebSocketManager()