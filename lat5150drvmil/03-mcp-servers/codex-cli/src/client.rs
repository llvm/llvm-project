//! Codex client implementation

use anyhow::{Context, Result};
use reqwest::{Client as HttpClient, header};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::config::CodexConfig;
use crate::streaming::ResponseStream;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    pub max_tokens: u32,
    pub temperature: f32,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_reasoning: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<Message>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,  // "user", "assistant", "developer"
    pub content: String,
}

impl Message {
    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }

    /// Create a developer message (system-level context)
    pub fn developer(content: impl Into<String>) -> Self {
        Self {
            role: "developer".to_string(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexResponse {
    pub content: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientStatus {
    pub model: String,
    pub authenticated: bool,
    pub message_count: usize,
}

pub struct CodexClient {
    config: CodexConfig,
    http_client: HttpClient,
    conversation_history: Vec<Message>,
}

impl CodexClient {
    /// Create new Codex client
    pub fn new(config: &CodexConfig) -> Result<Self> {
        info!("Initializing Codex client");

        if !config.is_authenticated() {
            warn!("Client not authenticated - some features may not work");
        }

        // Build HTTP client with appropriate headers
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            header::USER_AGENT,
            header::HeaderValue::from_static("codex-cli/0.1.0"),
        );

        // Add authentication headers
        match config.auth_method.as_str() {
            "chatgpt" => {
                if let Some(token) = &config.session_token {
                    let auth_value = format!("Bearer {}", token);
                    headers.insert(
                        header::AUTHORIZATION,
                        header::HeaderValue::from_str(&auth_value)?,
                    );
                }
            }
            "api_key" => {
                if let Some(key) = &config.api_key {
                    let auth_value = format!("Bearer {}", key);
                    headers.insert(
                        header::AUTHORIZATION,
                        header::HeaderValue::from_str(&auth_value)?,
                    );
                }
            }
            _ => {}
        }

        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .default_headers(headers)
            .build()?;

        Ok(Self {
            config: config.clone(),
            http_client,
            conversation_history: Vec::new(),
        })
    }

    /// Execute prompt and return complete response
    pub async fn execute(&self, prompt: &str) -> Result<CodexResponse> {
        debug!("Executing prompt: {}", prompt);

        let request = CodexRequest {
            model: self.config.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            enable_reasoning: Some(self.config.enable_reasoning),
            context: Some(self.conversation_history.clone()),
        };

        let response = self.http_client
            .post(&self.config.get_api_url())
            .json(&request)
            .send()
            .await
            .context("Failed to send request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API request failed ({}): {}", status, error_text);
        }

        let codex_response: CodexResponse = response
            .json()
            .await
            .context("Failed to parse response")?;

        debug!("Received response: {} chars", codex_response.content.len());

        Ok(codex_response)
    }

    /// Execute prompt with streaming response
    pub async fn execute_streaming(&self, prompt: &str) -> Result<ResponseStream> {
        debug!("Executing streaming prompt: {}", prompt);

        let request = CodexRequest {
            model: self.config.model.clone(),
            prompt: prompt.to_string(),
            stream: true,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            enable_reasoning: Some(self.config.enable_reasoning),
            context: Some(self.conversation_history.clone()),
        };

        let response = self.http_client
            .post(&self.config.get_api_url())
            .json(&request)
            .send()
            .await
            .context("Failed to send streaming request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API request failed ({}): {}", status, error_text);
        }

        ResponseStream::new(response).await
    }

    /// Get client status
    pub async fn status(&self) -> Result<ClientStatus> {
        Ok(ClientStatus {
            model: self.config.model.clone(),
            authenticated: self.config.is_authenticated(),
            message_count: self.conversation_history.len(),
        })
    }

    /// Clear conversation history
    pub async fn clear_history(&self) -> Result<()> {
        // Note: This is a placeholder - real implementation would need
        // mutable access or interior mutability
        info!("Clearing conversation history");
        Ok(())
    }

    /// Add message to conversation history
    pub fn add_to_history(&mut self, role: &str, content: &str) {
        self.conversation_history.push(Message {
            role: role.to_string(),
            content: content.to_string(),
        });
    }

    /// Get conversation history
    pub fn get_history(&self) -> &[Message] {
        &self.conversation_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let config = CodexConfig::default();
        let client = CodexClient::new(&config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_conversation_history() {
        let config = CodexConfig::default();
        let mut client = CodexClient::new(&config).unwrap();

        assert_eq!(client.get_history().len(), 0);

        client.add_to_history("user", "Hello");
        client.add_to_history("assistant", "Hi there!");

        assert_eq!(client.get_history().len(), 2);
        assert_eq!(client.get_history()[0].role, "user");
        assert_eq!(client.get_history()[1].content, "Hi there!");
    }
}
