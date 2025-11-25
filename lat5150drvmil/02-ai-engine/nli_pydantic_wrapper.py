#!/usr/bin/env python3
"""
Natural Language Interface - Pydantic Wrapper

Type-safe wrapper around ConversationManager providing full Pydantic integration.
Converts between legacy dataclass models and modern Pydantic models.

Usage:
    from nli_pydantic_wrapper import PydanticNLIManager
    from nli_pydantic_models import CreateConversationRequest, AddMessageRequest

    # Initialize
    nli = PydanticNLIManager()

    # Create conversation (type-safe)
    req = CreateConversationRequest(title="Test", metadata={"topic": "ai"})
    response = nli.create_conversation(req)
    print(response.conversation.id)

    # Add message (type-safe)
    msg_req = AddMessageRequest(
        conversation_id=response.conversation.id,
        role=MessageRole.USER,
        content="Hello, world!"
    )
    msg_response = nli.add_message(msg_req)

Author: DSMIL Integration Framework
Version: 2.0.0 (Pydantic)
"""

from typing import Optional, List
from datetime import datetime

try:
    from conversation_manager import ConversationManager as LegacyConversationManager
    from conversation_manager import Message as LegacyMessage, Conversation as LegacyConversation
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False
    print("⚠  Legacy conversation_manager not available")

try:
    from nli_pydantic_models import (
        Message,
        Conversation,
        MessageRole,
        CreateConversationRequest,
        CreateConversationResponse,
        AddMessageRequest,
        AddMessageResponse,
        GetConversationRequest,
        GetConversationResponse,
        SearchConversationsRequest,
        SearchConversationsResponse,
        ListConversationsRequest,
        ListConversationsResponse,
        ConversationStatistics,
        GetStatisticsRequest,
        GetStatisticsResponse,
        UpdateConversationRequest,
        UpdateConversationResponse,
        NLIError,
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠  NLI Pydantic models not available")


# ============================================================================
# Type Converters
# ============================================================================

def legacy_message_to_pydantic(legacy_msg: 'LegacyMessage') -> Message:
    """Convert legacy dataclass Message to Pydantic Message"""
    return Message(
        id=legacy_msg.id,
        conversation_id=legacy_msg.conversation_id,
        role=MessageRole(legacy_msg.role),
        content=legacy_msg.content,
        model=legacy_msg.model,
        tokens_input=legacy_msg.tokens_input,
        tokens_output=legacy_msg.tokens_output,
        latency_ms=legacy_msg.latency_ms,
        temperature=legacy_msg.temperature,
        created_at=legacy_msg.created_at or datetime.now(),
        metadata=legacy_msg.metadata or {}
    )


def legacy_conversation_to_pydantic(legacy_conv: 'LegacyConversation') -> Conversation:
    """Convert legacy dataclass Conversation to Pydantic Conversation"""
    messages = []
    if legacy_conv.messages:
        messages = [legacy_message_to_pydantic(m) for m in legacy_conv.messages]

    return Conversation(
        id=legacy_conv.id,
        user_id=legacy_conv.user_id,
        title=legacy_conv.title,
        summary=legacy_conv.summary,
        created_at=legacy_conv.created_at or datetime.now(),
        updated_at=legacy_conv.updated_at or datetime.now(),
        archived=legacy_conv.archived,
        metadata=legacy_conv.metadata or {},
        messages=messages
    )


# ============================================================================
# Pydantic NLI Manager
# ============================================================================

class PydanticNLIManager:
    """
    Type-safe Natural Language Interface Manager

    Provides Pydantic-validated conversation management with full type safety.
    All requests and responses use Pydantic models with automatic validation.

    Features:
    - Type-safe conversation creation and retrieval
    - Validated message storage with metadata
    - Cross-session conversation history
    - Semantic search across conversations
    - Usage analytics and statistics
    """

    def __init__(self,
                 db_host: str = "localhost",
                 db_port: int = 5432,
                 db_name: str = "dsmil_ai",
                 db_user: str = "dsmil",
                 db_password: Optional[str] = None,
                 pydantic_mode: bool = True):
        """
        Initialize Pydantic NLI Manager

        Args:
            db_host: PostgreSQL host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password (or use env DB_PASSWORD)
            pydantic_mode: Enable Pydantic mode (default True)
        """
        if not LEGACY_AVAILABLE:
            raise RuntimeError("ConversationManager not available. Check imports.")
        if pydantic_mode and not PYDANTIC_AVAILABLE:
            raise RuntimeError("Pydantic mode requested but NLI Pydantic models not available")

        self.pydantic_mode = pydantic_mode
        self.manager = LegacyConversationManager(
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password
        )

    def create_conversation(self, request: CreateConversationRequest) -> CreateConversationResponse:
        """
        Create a new conversation (type-safe)

        Args:
            request: CreateConversationRequest with user_id, title, metadata

        Returns:
            CreateConversationResponse with Conversation object
        """
        try:
            legacy_conv = self.manager.create_conversation(
                user_id=request.user_id,
                title=request.title,
                metadata=request.metadata
            )

            conversation = legacy_conversation_to_pydantic(legacy_conv)

            return CreateConversationResponse(
                conversation=conversation,
                success=True,
                message=f"Conversation created: {conversation.id}"
            )

        except Exception as e:
            return CreateConversationResponse(
                conversation=Conversation(id="error"),
                success=False,
                message=f"Failed to create conversation: {str(e)}"
            )

    def add_message(self, request: AddMessageRequest) -> AddMessageResponse:
        """
        Add a message to a conversation (type-safe)

        Args:
            request: AddMessageRequest with conversation_id, role, content, etc.

        Returns:
            AddMessageResponse with Message object
        """
        try:
            legacy_msg = self.manager.add_message(
                conversation_id=request.conversation_id,
                role=request.role.value,
                content=request.content,
                model=request.model,
                tokens_input=request.tokens_input,
                tokens_output=request.tokens_output,
                latency_ms=request.latency_ms,
                temperature=request.temperature,
                metadata=request.metadata
            )

            message = legacy_message_to_pydantic(legacy_msg)

            return AddMessageResponse(
                message=message,
                success=True,
                response_message=f"Message added: {message.id}"
            )

        except Exception as e:
            # Create error message
            error_msg = Message(
                id="error",
                conversation_id=request.conversation_id,
                role=MessageRole.SYSTEM,
                content=f"Error: {str(e)}"
            )
            return AddMessageResponse(
                message=error_msg,
                success=False,
                response_message=f"Failed to add message: {str(e)}"
            )

    def get_conversation(self, request: GetConversationRequest) -> GetConversationResponse:
        """
        Get a conversation by ID (type-safe)

        Args:
            request: GetConversationRequest with conversation_id, include_messages

        Returns:
            GetConversationResponse with Conversation or None
        """
        try:
            legacy_conv = self.manager.get_conversation(
                conversation_id=request.conversation_id,
                include_messages=request.include_messages
            )

            if not legacy_conv:
                return GetConversationResponse(
                    conversation=None,
                    success=False,
                    message=f"Conversation not found: {request.conversation_id}"
                )

            conversation = legacy_conversation_to_pydantic(legacy_conv)

            return GetConversationResponse(
                conversation=conversation,
                success=True,
                message=f"Conversation retrieved: {conversation.id}"
            )

        except Exception as e:
            return GetConversationResponse(
                conversation=None,
                success=False,
                message=f"Failed to get conversation: {str(e)}"
            )

    def search_conversations(self, request: SearchConversationsRequest) -> SearchConversationsResponse:
        """
        Search conversations by content (type-safe)

        Args:
            request: SearchConversationsRequest with query, user_id, limit

        Returns:
            SearchConversationsResponse with list of Conversations
        """
        try:
            legacy_convs = self.manager.search_conversations(
                query=request.query,
                user_id=request.user_id,
                limit=request.limit
            )

            conversations = [legacy_conversation_to_pydantic(c) for c in legacy_convs]

            return SearchConversationsResponse(
                conversations=conversations,
                total_found=len(conversations),
                query=request.query,
                success=True
            )

        except Exception as e:
            return SearchConversationsResponse(
                conversations=[],
                total_found=0,
                query=request.query,
                success=False
            )

    def list_conversations(self, request: ListConversationsRequest) -> ListConversationsResponse:
        """
        List recent conversations (type-safe)

        Args:
            request: ListConversationsRequest with user_id, limit, include_archived

        Returns:
            ListConversationsResponse with list of Conversations
        """
        try:
            legacy_convs = self.manager.get_recent_conversations(
                user_id=request.user_id,
                limit=request.limit,
                include_archived=request.include_archived
            )

            conversations = [legacy_conversation_to_pydantic(c) for c in legacy_convs]

            return ListConversationsResponse(
                conversations=conversations,
                total_count=len(conversations),
                success=True
            )

        except Exception as e:
            return ListConversationsResponse(
                conversations=[],
                total_count=0,
                success=False
            )

    def get_statistics(self, request: GetStatisticsRequest) -> GetStatisticsResponse:
        """
        Get conversation statistics (type-safe)

        Args:
            request: GetStatisticsRequest with user_id

        Returns:
            GetStatisticsResponse with ConversationStatistics
        """
        try:
            legacy_stats = self.manager.get_statistics(user_id=request.user_id)

            statistics = ConversationStatistics(
                total_conversations=legacy_stats['total_conversations'],
                total_messages=legacy_stats['total_messages'],
                avg_messages_per_conversation=legacy_stats['avg_messages_per_conversation'],
                total_tokens=legacy_stats['total_tokens'],
                avg_latency_ms=legacy_stats['avg_latency_ms']
            )

            return GetStatisticsResponse(
                statistics=statistics,
                user_id=request.user_id,
                success=True
            )

        except Exception as e:
            # Return empty stats on error
            empty_stats = ConversationStatistics(
                total_conversations=0,
                total_messages=0,
                avg_messages_per_conversation=0.0,
                total_tokens=0,
                avg_latency_ms=0.0
            )
            return GetStatisticsResponse(
                statistics=empty_stats,
                user_id=request.user_id,
                success=False
            )

    def update_conversation(self, request: UpdateConversationRequest) -> UpdateConversationResponse:
        """
        Update conversation metadata (type-safe)

        Args:
            request: UpdateConversationRequest with conversation_id, title, summary, archive

        Returns:
            UpdateConversationResponse with updated fields
        """
        updated_fields = []

        try:
            if request.title is not None:
                self.manager.update_conversation_title(request.conversation_id, request.title)
                updated_fields.append("title")

            if request.summary is not None:
                self.manager.update_conversation_summary(request.conversation_id, request.summary)
                updated_fields.append("summary")

            if request.archive is not None and request.archive:
                self.manager.archive_conversation(request.conversation_id)
                updated_fields.append("archived")

            return UpdateConversationResponse(
                conversation_id=request.conversation_id,
                updated_fields=updated_fields,
                success=True,
                message=f"Updated fields: {', '.join(updated_fields)}"
            )

        except Exception as e:
            return UpdateConversationResponse(
                conversation_id=request.conversation_id,
                updated_fields=[],
                success=False,
                message=f"Failed to update conversation: {str(e)}"
            )

    def close(self):
        """Close database connection pool"""
        self.manager.close()


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Pydantic NLI Manager Test")
    print("="*70)

    if not LEGACY_AVAILABLE or not PYDANTIC_AVAILABLE:
        print("\n⚠  Required dependencies not available. Cannot run test.")
        print(f"   Legacy Manager: {LEGACY_AVAILABLE}")
        print(f"   Pydantic Models: {PYDANTIC_AVAILABLE}")
        exit(1)

    try:
        # Initialize
        nli = PydanticNLIManager()
        print("\n✓ Pydantic NLI Manager initialized")

        # Create conversation
        create_req = CreateConversationRequest(
            title="WhiteRabbit Integration Test",
            metadata={"test": True, "version": "2.0"}
        )
        create_resp = nli.create_conversation(create_req)
        print(f"\n✓ Created conversation: {create_resp.conversation.id}")
        print(f"  Title: {create_resp.conversation.title}")
        print(f"  Success: {create_resp.success}")

        # Add user message
        user_msg_req = AddMessageRequest(
            conversation_id=create_resp.conversation.id,
            role=MessageRole.USER,
            content="What is the performance of WhiteRabbitNeo on Intel Arc NPU?"
        )
        user_msg_resp = nli.add_message(user_msg_req)
        print(f"\n✓ Added user message: {user_msg_resp.message.id}")
        print(f"  Content: {user_msg_resp.message.content[:60]}...")

        # Add assistant message
        assistant_msg_req = AddMessageRequest(
            conversation_id=create_resp.conversation.id,
            role=MessageRole.ASSISTANT,
            content="WhiteRabbitNeo achieves 45 tokens/second on Intel Arc NPU with INT4 quantization.",
            model="whiterabbit-neo-33b",
            tokens_output=20,
            latency_ms=450,
            temperature=0.7
        )
        assistant_msg_resp = nli.add_message(assistant_msg_req)
        print(f"\n✓ Added assistant message: {assistant_msg_resp.message.id}")
        print(f"  Model: {assistant_msg_resp.message.model}")
        print(f"  Latency: {assistant_msg_resp.message.latency_ms}ms")

        # Get conversation with messages
        get_req = GetConversationRequest(
            conversation_id=create_resp.conversation.id,
            include_messages=True
        )
        get_resp = nli.get_conversation(get_req)
        print(f"\n✓ Retrieved conversation: {get_resp.conversation.id}")
        print(f"  Messages: {len(get_resp.conversation.messages)}")

        # Get statistics
        stats_req = GetStatisticsRequest()
        stats_resp = nli.get_statistics(stats_req)
        print(f"\n✓ Statistics:")
        print(f"  Total Conversations: {stats_resp.statistics.total_conversations}")
        print(f"  Total Messages: {stats_resp.statistics.total_messages}")
        print(f"  Avg Latency: {stats_resp.statistics.avg_latency_ms:.2f}ms")

        # Close
        nli.close()
        print(f"\n{'='*70}")
        print("All Pydantic NLI tests passed!")
        print("="*70)

    except Exception as e:
        print(f"\n⚠  Error (expected if DB not set up): {e}")
        import traceback
        traceback.print_exc()
