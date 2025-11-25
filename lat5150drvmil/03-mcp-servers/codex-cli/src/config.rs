//! Configuration management for Codex CLI

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexConfig {
    /// Model to use (gpt-5-codex, gpt-5-codex-mini, etc.)
    pub model: String,

    /// Authentication method (chatgpt, api_key)
    pub auth_method: String,

    /// API base URL
    pub api_base: String,

    /// API endpoint for codex responses
    pub codex_endpoint: String,

    /// OpenAI API key (if using api_key method)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,

    /// Session token (if using chatgpt method)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_token: Option<String>,

    /// Request timeout in seconds
    pub timeout_seconds: u64,

    /// Enable extended thinking/reasoning
    pub enable_reasoning: bool,

    /// Maximum tokens per request
    pub max_tokens: u32,

    /// Temperature (0.0 - 2.0)
    pub temperature: f32,

    /// MCP server settings
    #[serde(default)]
    pub mcp: McpConfig,

    /// Custom prompts
    #[serde(default)]
    pub custom_prompts: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// Enable MCP server
    pub enabled: bool,

    /// MCP protocol version
    pub protocol_version: String,

    /// Available tools
    pub tools: Vec<String>,

    /// Server capabilities
    pub capabilities: Vec<String>,
}

impl Default for CodexConfig {
    fn default() -> Self {
        Self {
            model: "gpt-5-codex-mini".to_string(),
            auth_method: "chatgpt".to_string(),
            api_base: "https://chatgpt.com".to_string(),
            codex_endpoint: "/backend-api/codex/responses".to_string(),
            api_key: None,
            session_token: None,
            timeout_seconds: 300,
            enable_reasoning: true,
            max_tokens: 8192,
            temperature: 0.7,
            mcp: McpConfig::default(),
            custom_prompts: std::collections::HashMap::new(),
        }
    }
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            protocol_version: "2024-11-05".to_string(),
            tools: vec![
                "code_generation".to_string(),
                "code_review".to_string(),
                "debugging".to_string(),
                "refactoring".to_string(),
                "documentation".to_string(),
            ],
            capabilities: vec![
                "streaming".to_string(),
                "reasoning".to_string(),
                "code_context".to_string(),
            ],
        }
    }
}

impl CodexConfig {
    /// Load configuration from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(path.as_ref())
            .context("Failed to read configuration file")?;

        let config: Self = toml::from_str(&contents)
            .context("Failed to parse configuration file")?;

        Ok(config)
    }

    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents = toml::to_string_pretty(self)
            .context("Failed to serialize configuration")?;

        if let Some(parent) = path.as_ref().parent() {
            fs::create_dir_all(parent)
                .context("Failed to create configuration directory")?;
        }

        fs::write(path.as_ref(), contents)
            .context("Failed to write configuration file")?;

        Ok(())
    }

    /// Check if authenticated
    pub fn is_authenticated(&self) -> bool {
        match self.auth_method.as_str() {
            "chatgpt" => self.session_token.is_some(),
            "api_key" => self.api_key.is_some(),
            _ => false,
        }
    }

    /// Set configuration value by key
    pub fn set_value(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "model" => self.model = value.to_string(),
            "auth_method" => self.auth_method = value.to_string(),
            "api_base" => self.api_base = value.to_string(),
            "enable_reasoning" => {
                self.enable_reasoning = value.parse()
                    .context("Invalid boolean value for enable_reasoning")?;
            }
            "temperature" => {
                self.temperature = value.parse()
                    .context("Invalid float value for temperature")?;
            }
            "max_tokens" => {
                self.max_tokens = value.parse()
                    .context("Invalid integer value for max_tokens")?;
            }
            _ => anyhow::bail!("Unknown configuration key: {}", key),
        }

        Ok(())
    }

    /// Get full API URL
    pub fn get_api_url(&self) -> String {
        format!("{}{}", self.api_base, self.codex_endpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = CodexConfig::default();
        assert_eq!(config.model, "gpt-5-codex-mini");
        assert_eq!(config.auth_method, "chatgpt");
        assert!(!config.is_authenticated());
    }

    #[test]
    fn test_save_and_load() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = CodexConfig::default();

        config.save(temp_file.path()).unwrap();
        let loaded = CodexConfig::load(temp_file.path()).unwrap();

        assert_eq!(config.model, loaded.model);
        assert_eq!(config.auth_method, loaded.auth_method);
    }

    #[test]
    fn test_authentication_check() {
        let mut config = CodexConfig::default();
        assert!(!config.is_authenticated());

        config.session_token = Some("test_token".to_string());
        assert!(config.is_authenticated());

        config.auth_method = "api_key".to_string();
        config.session_token = None;
        assert!(!config.is_authenticated());

        config.api_key = Some("sk-test".to_string());
        assert!(config.is_authenticated());
    }
}
