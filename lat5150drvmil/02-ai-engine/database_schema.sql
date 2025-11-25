-- LAT5150DRVMIL AI Database Schema
-- PostgreSQL Schema for Conversation History, User Preferences, and Analytics
-- Author: DSMIL Integration Framework
-- Version: 1.0.0

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- USERS & PROFILES
-- ============================================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    preference_key VARCHAR(255) NOT NULL,
    preference_value JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, preference_key)
);

-- ============================================================================
-- CONVERSATIONS & MESSAGES
-- ============================================================================

CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    title VARCHAR(500),
    summary TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    archived BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    model VARCHAR(100),
    tokens_input INT,
    tokens_output INT,
    latency_ms INT,
    temperature FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Message embeddings for semantic search across conversation history
CREATE TABLE message_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    embedding_model VARCHAR(100) NOT NULL,
    embedding vector(384), -- sentence-transformers/all-MiniLM-L6-v2 produces 384-dim vectors
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- RAG SYSTEM
-- ============================================================================

CREATE TABLE rag_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_hash VARCHAR(64) UNIQUE NOT NULL,
    file_path TEXT,
    file_type VARCHAR(50),
    title VARCHAR(500),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE rag_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES rag_documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

CREATE TABLE rag_chunk_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID REFERENCES rag_chunks(id) ON DELETE CASCADE,
    embedding_model VARCHAR(100) NOT NULL,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- RAG RETRIEVALS (What was retrieved for each query)
-- ============================================================================

CREATE TABLE rag_retrievals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES rag_chunks(id) ON DELETE SET NULL,
    relevance_score FLOAT,
    rank INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- KNOWLEDGE GRAPH (Enhanced from JSON-based system)
-- ============================================================================

CREATE TABLE kg_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id VARCHAR(16) UNIQUE NOT NULL, -- Original hash-based ID
    name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE kg_observations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_id UUID REFERENCES kg_entities(id) ON DELETE CASCADE,
    observation TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE kg_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_entity_id UUID REFERENCES kg_entities(id) ON DELETE CASCADE,
    to_entity_id UUID REFERENCES kg_entities(id) ON DELETE CASCADE,
    relation_type VARCHAR(100) NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- QUERY ANALYTICS
-- ============================================================================

CREATE TABLE query_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    query_hash VARCHAR(64),
    query_text TEXT,
    model VARCHAR(100),
    tokens_total INT,
    latency_ms INT,
    cache_hit BOOLEAN DEFAULT FALSE,
    rag_enabled BOOLEAN DEFAULT FALSE,
    rag_chunks_retrieved INT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- ============================================================================
-- RESPONSE CACHE (for Redis backup/persistence)
-- ============================================================================

CREATE TABLE response_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(64) UNIQUE NOT NULL,
    query_hash VARCHAR(64),
    model VARCHAR(100),
    response TEXT NOT NULL,
    tokens INT,
    created_at TIMESTAMP DEFAULT NOW(),
    accessed_at TIMESTAMP DEFAULT NOW(),
    access_count INT DEFAULT 0,
    ttl_seconds INT DEFAULT 3600,
    expires_at TIMESTAMP
);

-- ============================================================================
-- MODEL PERFORMANCE METRICS
-- ============================================================================

CREATE TABLE model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- 'latency', 'tokens', 'error_rate', etc.
    metric_value FLOAT NOT NULL,
    sample_size INT DEFAULT 1,
    recorded_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- ============================================================================
-- HIERARCHICAL MEMORY (Working/Short-Term/Long-Term)
-- ============================================================================

CREATE TABLE memory_blocks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    block_id VARCHAR(16) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    token_count INT NOT NULL,
    block_type VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system', 'tool_result', etc.
    priority INT DEFAULT 5,
    tier VARCHAR(20) NOT NULL, -- 'working', 'short_term', 'long_term'
    created_at TIMESTAMP DEFAULT NOW(),
    accessed_at TIMESTAMP DEFAULT NOW(),
    access_count INT DEFAULT 0,
    summary TEXT,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    phase VARCHAR(50),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_memory_blocks_block_id ON memory_blocks(block_id);
CREATE INDEX idx_memory_blocks_tier ON memory_blocks(tier);
CREATE INDEX idx_memory_blocks_conversation_id ON memory_blocks(conversation_id);
CREATE INDEX idx_memory_blocks_accessed_at ON memory_blocks(accessed_at DESC);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Conversations
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at DESC);
CREATE INDEX idx_conversations_archived ON conversations(archived) WHERE archived = FALSE;

-- Messages
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX idx_messages_role ON messages(role);

-- RAG
CREATE INDEX idx_rag_documents_hash ON rag_documents(document_hash);
CREATE INDEX idx_rag_chunks_document_id ON rag_chunks(document_id);
CREATE INDEX idx_rag_retrievals_message_id ON rag_retrievals(message_id);

-- Knowledge Graph
CREATE INDEX idx_kg_entities_entity_id ON kg_entities(entity_id);
CREATE INDEX idx_kg_entities_type ON kg_entities(entity_type);
CREATE INDEX idx_kg_observations_entity_id ON kg_observations(entity_id);
CREATE INDEX idx_kg_relations_from_to ON kg_relations(from_entity_id, to_entity_id);

-- Analytics
CREATE INDEX idx_query_analytics_user_id ON query_analytics(user_id);
CREATE INDEX idx_query_analytics_created_at ON query_analytics(created_at DESC);
CREATE INDEX idx_query_analytics_cache_hit ON query_analytics(cache_hit);

-- Cache
CREATE INDEX idx_response_cache_key ON response_cache(cache_key);
CREATE INDEX idx_response_cache_expires_at ON response_cache(expires_at);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Recent conversations with message counts
CREATE VIEW recent_conversations AS
SELECT
    c.id,
    c.user_id,
    c.title,
    c.summary,
    c.created_at,
    c.updated_at,
    COUNT(m.id) as message_count,
    MAX(m.created_at) as last_message_at
FROM conversations c
LEFT JOIN messages m ON c.id = m.conversation_id
WHERE c.archived = FALSE
GROUP BY c.id, c.user_id, c.title, c.summary, c.created_at, c.updated_at
ORDER BY last_message_at DESC;

-- Model performance summary
CREATE VIEW model_performance AS
SELECT
    model,
    COUNT(*) as total_queries,
    AVG(latency_ms) as avg_latency_ms,
    AVG(tokens_total) as avg_tokens,
    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as cache_hit_rate,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
FROM query_analytics
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY model;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rag_documents_updated_at BEFORE UPDATE ON rag_documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_kg_entities_updated_at BEFORE UPDATE ON kg_entities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update cache accessed_at and access_count
CREATE OR REPLACE FUNCTION update_cache_access()
RETURNS TRIGGER AS $$
BEGIN
    NEW.accessed_at = NOW();
    NEW.access_count = NEW.access_count + 1;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_response_cache_access BEFORE UPDATE ON response_cache
    FOR EACH ROW EXECUTE FUNCTION update_cache_access();

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Create default user
INSERT INTO users (username, email, metadata) VALUES
('default', 'user@localhost', '{"system": "LAT5150DRVMIL", "created_by": "setup_script"}'::jsonb)
ON CONFLICT (username) DO NOTHING;

-- ============================================================================
-- CLEANUP FUNCTIONS
-- ============================================================================

-- Clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM response_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Archive old conversations
CREATE OR REPLACE FUNCTION archive_old_conversations(days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    UPDATE conversations
    SET archived = TRUE
    WHERE updated_at < NOW() - (days || ' days')::INTERVAL
    AND archived = FALSE;
    GET DIAGNOSTICS archived_count = ROW_COUNT;
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- GRANTS (Adjust as needed for your user)
-- ============================================================================

-- Grant all privileges to dsmil user (create this user first)
-- CREATE USER dsmil WITH PASSWORD 'your_secure_password';
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dsmil;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dsmil;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO dsmil;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE users IS 'User accounts and profiles';
COMMENT ON TABLE conversations IS 'Conversation sessions with complete history';
COMMENT ON TABLE messages IS 'Individual messages within conversations';
COMMENT ON TABLE message_embeddings IS 'Vector embeddings for semantic search across messages';
COMMENT ON TABLE rag_documents IS 'Documents indexed for RAG retrieval';
COMMENT ON TABLE rag_chunks IS 'Document chunks for better retrieval granularity';
COMMENT ON TABLE rag_chunk_embeddings IS 'Vector embeddings for semantic document search';
COMMENT ON TABLE query_analytics IS 'Query performance and usage analytics';
COMMENT ON TABLE response_cache IS 'Persistent cache for frequently asked queries';
COMMENT ON TABLE model_metrics IS 'Model performance metrics over time';

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Show all tables
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

-- Show all indexes
SELECT indexname, tablename
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
