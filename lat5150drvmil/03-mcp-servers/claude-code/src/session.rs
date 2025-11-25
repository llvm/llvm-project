//! Session management

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub message_count: usize,
}

pub struct SessionManager {}

impl SessionManager {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    pub async fn create_session(&self, name: Option<&str>) -> Result<Session> {
        Ok(Session {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.unwrap_or("default").to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            message_count: 0,
        })
    }

    pub async fn list_sessions(&self) -> Result<Vec<Session>> {
        Ok(vec![])
    }

    pub async fn load_session(&self, session_id: &str) -> Result<Session> {
        Ok(Session {
            id: session_id.to_string(),
            name: "loaded".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            message_count: 0,
        })
    }

    pub async fn delete_session(&self, _session_id: &str) -> Result<()> {
        Ok(())
    }
}
