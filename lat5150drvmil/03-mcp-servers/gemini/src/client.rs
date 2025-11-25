//! Gemini API client implementation

use anyhow::{Context, Result};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tracing::{debug, info};

use crate::config::GeminiConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiRequest {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<Vec<SafetySetting>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    pub role: String,
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    Text { text: String },
    InlineData { inline_data: InlineData },
    FunctionCall { function_call: FunctionCall },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineData {
    pub mime_type: String,
    pub data: String,  // base64 encoded
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    pub category: String,
    pub threshold: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_declarations: Option<Vec<FunctionDeclaration>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub google_search_retrieval: Option<GoogleSearchRetrieval>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_execution: Option<CodeExecution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleSearchRetrieval {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecution {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiResponse {
    pub candidates: Vec<Candidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetadata {
    pub prompt_token_count: usize,
    pub candidates_token_count: usize,
    pub total_token_count: usize,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub function_calls: Option<Vec<FunctionCall>>,
    pub executed_code: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SessionStats {
    pub message_count: usize,
    pub total_tokens: usize,
    pub thinking_enabled: bool,
    pub grounding_enabled: bool,
}

pub struct GeminiClient {
    config: GeminiConfig,
    http_client: HttpClient,
    conversation_history: Vec<Content>,
    thinking_mode: bool,
    grounding_enabled: bool,
    code_execution_enabled: bool,
    total_tokens: usize,
}

impl GeminiClient {
    pub async fn new(config: &GeminiConfig) -> Result<Self> {
        info!("Initializing Gemini client");

        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()?;

        Ok(Self {
            config: config.clone(),
            http_client,
            conversation_history: Vec::new(),
            thinking_mode: config.default_thinking,
            grounding_enabled: config.default_grounding,
            code_execution_enabled: config.default_code_execution,
            total_tokens: 0,
        })
    }

    pub async fn generate(&mut self, prompt: &str) -> Result<GenerateResponse> {
        debug!("Generating response for: {}", prompt);

        // Add user message to history
        self.conversation_history.push(Content {
            role: "user".to_string(),
            parts: vec![Part::Text { text: prompt.to_string() }],
        });

        let request = self.build_request()?;

        let api_key = self.config.api_key.as_ref()
            .context("API key not configured")?;

        let model_name = if self.thinking_mode {
            "gemini-2.0-flash-thinking-exp"
        } else {
            &self.config.model
        };

        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.config.api_base,
            model_name,
            api_key
        );

        let response = self.http_client
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("Failed to send request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("API request failed ({}): {}", status, error_text);
        }

        let gemini_response: GeminiResponse = response
            .json()
            .await
            .context("Failed to parse response")?;

        // Update token count
        if let Some(usage) = &gemini_response.usage_metadata {
            self.total_tokens += usage.total_token_count;
        }

        // Extract response
        let text = gemini_response.candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .and_then(|p| match p {
                Part::Text { text } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();

        // Add assistant response to history
        if let Some(candidate) = gemini_response.candidates.first() {
            self.conversation_history.push(candidate.content.clone());
        }

        Ok(GenerateResponse {
            text,
            function_calls: None,
            executed_code: None,
        })
    }

    pub async fn generate_streaming(&self, _prompt: &str) -> Result<ResponseStream> {
        // Stub implementation
        Ok(ResponseStream {})
    }

    pub async fn generate_multimodal(&self, _prompt: &str, _files: &[impl AsRef<Path>]) -> Result<GenerateResponse> {
        // Stub implementation
        Ok(GenerateResponse {
            text: "Multimodal response (stub)".to_string(),
            function_calls: None,
            executed_code: None,
        })
    }

    pub async fn generate_with_functions(&self, _prompt: &str, _functions: &[FunctionDeclaration]) -> Result<GenerateResponse> {
        // Stub implementation
        Ok(GenerateResponse {
            text: "Function calling response (stub)".to_string(),
            function_calls: None,
            executed_code: None,
        })
    }

    pub async fn analyze_image(&self, _path: &str, _question: &str) -> Result<()> {
        println!("Image analysis (stub)");
        Ok(())
    }

    pub async fn analyze_video(&self, _path: &str, _question: &str) -> Result<()> {
        println!("Video analysis (stub)");
        Ok(())
    }

    pub async fn transcribe_audio(&self, _path: &str) -> Result<()> {
        println!("Audio transcription (stub)");
        Ok(())
    }

    pub fn enable_thinking_mode(&mut self) -> Result<()> {
        self.thinking_mode = true;
        Ok(())
    }

    pub fn toggle_thinking_mode(&mut self) -> Result<()> {
        self.thinking_mode = !self.thinking_mode;
        Ok(())
    }

    pub fn enable_grounding(&mut self) -> Result<()> {
        self.grounding_enabled = true;
        Ok(())
    }

    pub fn toggle_grounding(&mut self) -> Result<()> {
        self.grounding_enabled = !self.grounding_enabled;
        Ok(())
    }

    pub fn enable_code_execution(&mut self) -> Result<()> {
        self.code_execution_enabled = true;
        Ok(())
    }

    pub fn clear_history(&mut self) -> Result<()> {
        self.conversation_history.clear();
        Ok(())
    }

    pub fn get_stats(&self) -> SessionStats {
        SessionStats {
            message_count: self.conversation_history.len(),
            total_tokens: self.total_tokens,
            thinking_enabled: self.thinking_mode,
            grounding_enabled: self.grounding_enabled,
        }
    }

    fn build_request(&self) -> Result<GeminiRequest> {
        let mut tools = Vec::new();

        if self.grounding_enabled {
            tools.push(Tool {
                google_search_retrieval: Some(GoogleSearchRetrieval {}),
                function_declarations: None,
                code_execution: None,
            });
        }

        if self.code_execution_enabled {
            tools.push(Tool {
                code_execution: Some(CodeExecution {}),
                function_declarations: None,
                google_search_retrieval: None,
            });
        }

        Ok(GeminiRequest {
            contents: self.conversation_history.clone(),
            generation_config: Some(GenerationConfig {
                temperature: self.config.temperature,
                top_p: self.config.top_p,
                top_k: self.config.top_k,
                max_output_tokens: self.config.max_output_tokens,
            }),
            safety_settings: None,
            tools: if tools.is_empty() { None } else { Some(tools) },
        })
    }
}

pub struct ResponseStream {}

impl ResponseStream {
    pub async fn next(&mut self) -> Option<Result<String>> {
        None
    }
}
