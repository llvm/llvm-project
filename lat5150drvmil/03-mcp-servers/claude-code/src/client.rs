//! Claude Code API client implementation

use anyhow::{Context, Result};
use reqwest::{Client as HttpClient, header};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, info};

use crate::config::ClaudeCodeConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeResponse {
    pub content: String,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub name: String,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Stats {
    pub message_count: usize,
    pub avg_latency_ms: f64,
    pub total_tokens: usize,
}

pub struct ClaudeCodeClient {
    config: ClaudeCodeConfig,
    http_client: HttpClient,
    conversation_history: Vec<Message>,
}

impl ClaudeCodeClient {
    pub async fn new(config: &ClaudeCodeConfig) -> Result<Self> {
        info!("Initializing Claude Code client");

        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            "anthropic-version",
            header::HeaderValue::from_static("2023-06-01"),
        );

        if let Some(token) = &config.api.auth_token {
            headers.insert(
                "x-api-key",
                header::HeaderValue::from_str(token)?,
            );
        }

        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(config.api.timeout_seconds))
            .default_headers(headers)
            .build()?;

        Ok(Self {
            config: config.clone(),
            http_client,
            conversation_history: Vec::new(),
        })
    }

    pub async fn execute(&self, prompt: &str) -> Result<ClaudeResponse> {
        debug!("Executing prompt: {}", prompt);

        let mut messages = self.conversation_history.clone();
        messages.push(Message {
            role: "user".to_string(),
            content: prompt.to_string(),
        });

        let request = ClaudeRequest {
            model: self.config.api.model.clone(),
            messages,
            max_tokens: self.config.api.max_tokens,
            temperature: self.config.api.temperature,
            stream: false,
        };

        let response = self.http_client
            .post(format!("{}/messages", self.config.api.base_url))
            .json(&request)
            .send()
            .await
            .context("Failed to send request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API request failed ({}): {}", status, error_text);
        }

        let claude_response: ClaudeResponse = response
            .json()
            .await
            .context("Failed to parse response")?;

        Ok(claude_response)
    }

    pub async fn execute_streaming(&self, _prompt: &str) -> Result<ResponseStream> {
        // Stub implementation
        Ok(ResponseStream {})
    }

    pub async fn list_agents(&self) -> Result<Vec<AgentInfo>> {
        // Stub implementation
        Ok(vec![
            AgentInfo {
                name: "Code Generator".to_string(),
                capabilities: vec!["generation".to_string()],
            },
        ])
    }

    pub async fn get_stats(&self) -> Result<Stats> {
        // Stub implementation
        Ok(Stats {
            message_count: self.conversation_history.len(),
            avg_latency_ms: 0.0,
            total_tokens: 0,
        })
    }
}

pub struct ResponseStream {}

impl ResponseStream {
    pub async fn next(&mut self) -> Option<Result<String>> {
        None
    }
}
