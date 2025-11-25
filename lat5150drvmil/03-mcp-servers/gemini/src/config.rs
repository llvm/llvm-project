//! Configuration management for Gemini CLI

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiConfig {
    /// API key
    pub api_key: Option<String>,

    /// Model to use
    pub model: String,

    /// API base URL
    pub api_base: String,

    /// Request timeout (seconds)
    pub timeout_seconds: u64,

    /// Max output tokens
    pub max_output_tokens: u32,

    /// Temperature (0.0 - 2.0)
    pub temperature: f32,

    /// Top-P sampling
    pub top_p: f32,

    /// Top-K sampling
    pub top_k: u32,

    /// Enable thinking mode by default
    pub default_thinking: bool,

    /// Enable grounding by default
    pub default_grounding: bool,

    /// Enable code execution by default
    pub default_code_execution: bool,

    /// Safety settings
    pub safety: SafetyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    /// Block harassment
    pub block_harassment: String,

    /// Block hate speech
    pub block_hate_speech: String,

    /// Block sexually explicit
    pub block_sexually_explicit: String,

    /// Block dangerous content
    pub block_dangerous_content: String,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            model: "gemini-2.0-flash-exp".to_string(),
            api_base: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            timeout_seconds: 300,
            max_output_tokens: 8192,
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            default_thinking: false,
            default_grounding: false,
            default_code_execution: false,
            safety: SafetyConfig::default(),
        }
    }
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            block_harassment: "BLOCK_MEDIUM_AND_ABOVE".to_string(),
            block_hate_speech: "BLOCK_MEDIUM_AND_ABOVE".to_string(),
            block_sexually_explicit: "BLOCK_MEDIUM_AND_ABOVE".to_string(),
            block_dangerous_content: "BLOCK_MEDIUM_AND_ABOVE".to_string(),
        }
    }
}

impl GeminiConfig {
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

    /// Set configuration value by key
    pub fn set_value(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "model" => self.model = value.to_string(),
            "api_base" => self.api_base = value.to_string(),
            "temperature" => {
                self.temperature = value.parse()
                    .context("Invalid float value")?;
            }
            "max_output_tokens" => {
                self.max_output_tokens = value.parse()
                    .context("Invalid integer value")?;
            }
            "default_thinking" => {
                self.default_thinking = value.parse()
                    .context("Invalid boolean value")?;
            }
            "default_grounding" => {
                self.default_grounding = value.parse()
                    .context("Invalid boolean value")?;
            }
            _ => anyhow::bail!("Unknown configuration key: {}", key),
        }

        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_output_tokens == 0 {
            anyhow::bail!("max_output_tokens must be greater than 0");
        }

        if !(0.0..=2.0).contains(&self.temperature) {
            anyhow::bail!("temperature must be between 0.0 and 2.0");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = GeminiConfig::default();
        assert_eq!(config.model, "gemini-2.0-flash-exp");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_save_and_load() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = GeminiConfig::default();

        config.save(temp_file.path()).unwrap();
        let loaded = GeminiConfig::load(temp_file.path()).unwrap();

        assert_eq!(config.model, loaded.model);
    }
}
