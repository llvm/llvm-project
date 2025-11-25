#!/usr/bin/env python3
"""
Natural Language Interface - Pydantic Models

Type-safe models for conversational AI with persistent history.
Provides Pydantic wrappers for the ConversationManager with full validation.

Author: DSMIL Integration Framework
Version: 2.0.0 (Pydantic)
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class MessageRole(str, Enum):
    """Role of message sender"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


# ============================================================================
# Core Models
# ============================================================================

class Message(BaseModel):
    """Type-safe message representation"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=100000)
    model: Optional[str] = Field(None, description="AI model used (e.g., whiterabbit-neo-33b)")
    tokens_input: Optional[int] = Field(None, ge=0, description="Input token count")
    tokens_output: Optional[int] = Field(None, ge=0, description="Output token count")
    latency_ms: Optional[int] = Field(None, ge=0, description="Response latency in milliseconds")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Ensure content is not just whitespace"""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Conversation(BaseModel):
    """Type-safe conversation representation"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = Field(None, description="User who owns this conversation")
    title: Optional[str] = Field(None, max_length=500, description="Conversation title")
    summary: Optional[str] = Field(None, max_length=2000, description="Auto-generated summary")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    archived: bool = Field(default=False, description="Whether conversation is archived")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateConversationRequest(BaseModel):
    """Request to create a new conversation"""
    user_id: Optional[str] = None
    title: Optional[str] = Field(None, max_length=500)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateConversationResponse(BaseModel):
    """Response from creating a conversation"""
    conversation: Conversation
    success: bool = True
    message: str = "Conversation created successfully"


class AddMessageRequest(BaseModel):
    """Request to add a message to a conversation"""
    conversation_id: str
    role: MessageRole
    content: str = Field(..., min_length=1, max_length=100000)
    model: Optional[str] = None
    tokens_input: Optional[int] = Field(None, ge=0)
    tokens_output: Optional[int] = Field(None, ge=0)
    latency_ms: Optional[int] = Field(None, ge=0)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AddMessageResponse(BaseModel):
    """Response from adding a message"""
    message: Message
    success: bool = True
    response_message: str = "Message added successfully"


class GetConversationRequest(BaseModel):
    """Request to retrieve a conversation"""
    conversation_id: str
    include_messages: bool = Field(default=True, description="Include message history")


class GetConversationResponse(BaseModel):
    """Response from getting a conversation"""
    conversation: Optional[Conversation]
    success: bool = True
    message: str = "Conversation retrieved successfully"


class SearchConversationsRequest(BaseModel):
    """Request to search conversations"""
    query: str = Field(..., min_length=1, max_length=500)
    user_id: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)


class SearchConversationsResponse(BaseModel):
    """Response from searching conversations"""
    conversations: List[Conversation]
    total_found: int
    query: str
    success: bool = True


class ListConversationsRequest(BaseModel):
    """Request to list recent conversations"""
    user_id: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)
    include_archived: bool = Field(default=False)


class ListConversationsResponse(BaseModel):
    """Response from listing conversations"""
    conversations: List[Conversation]
    total_count: int
    success: bool = True


class ConversationStatistics(BaseModel):
    """Conversation usage statistics"""
    total_conversations: int = Field(ge=0)
    total_messages: int = Field(ge=0)
    avg_messages_per_conversation: float = Field(ge=0.0)
    total_tokens: int = Field(ge=0)
    avg_latency_ms: float = Field(ge=0.0)


class GetStatisticsRequest(BaseModel):
    """Request for conversation statistics"""
    user_id: Optional[str] = None


class GetStatisticsResponse(BaseModel):
    """Response with conversation statistics"""
    statistics: ConversationStatistics
    user_id: Optional[str]
    success: bool = True


class UpdateConversationRequest(BaseModel):
    """Request to update conversation metadata"""
    conversation_id: str
    title: Optional[str] = Field(None, max_length=500)
    summary: Optional[str] = Field(None, max_length=2000)
    archive: Optional[bool] = None


class UpdateConversationResponse(BaseModel):
    """Response from updating conversation"""
    conversation_id: str
    updated_fields: List[str]
    success: bool = True
    message: str = "Conversation updated successfully"


# ============================================================================
# NLI Chat Models (Integration with Orchestrator)
# ============================================================================

class NLIChatRequest(BaseModel):
    """Type-safe request for NLI chat with conversation history"""
    prompt: str = Field(..., min_length=1, max_length=32000)
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    user_id: Optional[str] = None
    model: Optional[str] = Field(None, description="Force specific model")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32000)
    include_history: bool = Field(default=True, description="Include conversation history")
    history_limit: int = Field(default=10, ge=1, le=50, description="Number of previous messages to include")
    create_new_conversation: bool = Field(default=False, description="Force new conversation")
    conversation_title: Optional[str] = Field(None, max_length=500)


class NLIChatResponse(BaseModel):
    """Type-safe response from NLI chat"""
    response: str = Field(..., min_length=1)
    conversation_id: str
    message_id: str
    model_used: str
    backend: str
    tokens_input: int = Field(ge=0)
    tokens_output: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    temperature: float = Field(ge=0.0, le=2.0)
    conversation_history: List[Message] = Field(default_factory=list)
    created_new_conversation: bool = Field(default=False)
    success: bool = True


# ============================================================================
# Error Models
# ============================================================================

class NLIError(BaseModel):
    """Error response from NLI operations"""
    success: bool = False
    error: str
    error_type: Literal["validation", "database", "not_found", "internal"]
    conversation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("NLI Pydantic Models - Type-Safe Conversational AI")
    print("="*70)

    # Example: Create conversation request
    create_req = CreateConversationRequest(
        title="WhiteRabbit Integration Discussion",
        metadata={"topic": "pydantic_integration", "priority": "high"}
    )
    print(f"\n1. Create Conversation Request:")
    print(f"   {create_req.model_dump_json(indent=2)}")

    # Example: Add message request
    add_msg_req = AddMessageRequest(
        conversation_id="test-conv-123",
        role=MessageRole.USER,
        content="What are the benefits of WhiteRabbitNeo over DeepSeek?",
        model="whiterabbit-neo-33b",
        temperature=0.7
    )
    print(f"\n2. Add Message Request:")
    print(f"   Role: {add_msg_req.role.value}")
    print(f"   Content: {add_msg_req.content[:60]}...")

    # Example: NLI chat request
    nli_req = NLIChatRequest(
        prompt="Explain multi-device inference in WhiteRabbit",
        include_history=True,
        history_limit=5,
        temperature=0.8
    )
    print(f"\n3. NLI Chat Request:")
    print(f"   Prompt: {nli_req.prompt}")
    print(f"   Include History: {nli_req.include_history}")
    print(f"   History Limit: {nli_req.history_limit}")

    # Example: Message with validation
    try:
        invalid_msg = Message(
            conversation_id="test",
            role=MessageRole.USER,
            content="   ",  # Empty content
        )
    except Exception as e:
        print(f"\n4. Validation Error (expected):")
        print(f"   {e}")

    # Example: Valid message
    valid_msg = Message(
        conversation_id="conv-456",
        role=MessageRole.ASSISTANT,
        content="WhiteRabbitNeo supports NPU, GPU (Arc), NCS2, and CPU backends.",
        model="whiterabbit-neo-33b",
        tokens_output=20,
        latency_ms=450,
        temperature=0.7
    )
    print(f"\n5. Valid Message:")
    print(f"   ID: {valid_msg.id}")
    print(f"   Role: {valid_msg.role.value}")
    print(f"   Tokens: {valid_msg.tokens_output}")

    print(f"\n{'='*70}")
    print("All models validated successfully!")
    print("="*70)
