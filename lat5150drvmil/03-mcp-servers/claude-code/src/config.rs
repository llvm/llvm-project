//! Configuration management for Claude Code client

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeCodeConfig {
    /// API configuration
    pub api: ApiConfig,

    /// Hardware acceleration settings
    pub hardware: HardwareConfig,

    /// Agent orchestration settings
    pub agents: AgentConfig,

    /// Git analysis settings
    pub git: GitConfig,

    /// Session management
    pub session: SessionConfig,

    /// Performance settings
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// API base URL
    pub base_url: String,

    /// API key or session token
    pub auth_token: Option<String>,

    /// Request timeout (seconds)
    pub timeout_seconds: u64,

    /// Max tokens per request
    pub max_tokens: u32,

    /// Temperature
    pub temperature: f32,

    /// Model to use
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Enable NPU acceleration (Intel AI Boost)
    pub enable_npu: bool,

    /// Enable AVX-512 SIMD
    pub enable_avx512: bool,

    /// Enable GPU acceleration (OpenVINO)
    pub enable_gpu: bool,

    /// NPU device index
    pub npu_device: usize,

    /// Number of P-cores to use
    pub p_cores: usize,

    /// Number of E-cores to use
    pub e_cores: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Enable agent orchestration
    pub enable_orchestration: bool,

    /// Max concurrent agents
    pub max_concurrent: usize,

    /// Agent timeout (seconds)
    pub agent_timeout: u64,

    /// Use binary IPC
    pub use_binary_ipc: bool,

    /// Shared memory size (bytes)
    pub shm_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitConfig {
    /// Enable ShadowGit analysis
    pub enable_analysis: bool,

    /// Enable conflict prediction
    pub enable_conflict_prediction: bool,

    /// Enable fast diff with SIMD
    pub enable_fast_diff: bool,

    /// Max repository size (MB)
    pub max_repo_size_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Auto-save sessions
    pub auto_save: bool,

    /// Session save directory
    pub save_dir: String,

    /// Max sessions to keep
    pub max_sessions: usize,

    /// Session timeout (hours)
    pub session_timeout_hours: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,

    /// Metrics export port
    pub metrics_port: u16,

    /// Enable tracing
    pub enable_tracing: bool,

    /// Log level
    pub log_level: String,
}

impl Default for ClaudeCodeConfig {
    fn default() -> Self {
        Self {
            api: ApiConfig {
                base_url: "https://api.anthropic.com/v1".to_string(),
                auth_token: None,
                timeout_seconds: 300,
                max_tokens: 8192,
                temperature: 0.7,
                model: "claude-3-5-sonnet-20241022".to_string(),
            },
            hardware: HardwareConfig {
                enable_npu: false,
                enable_avx512: false,
                enable_gpu: false,
                npu_device: 0,
                p_cores: 6,
                e_cores: 10,
            },
            agents: AgentConfig {
                enable_orchestration: true,
                max_concurrent: 4,
                agent_timeout: 120,
                use_binary_ipc: true,
                shm_size: 1024 * 1024 * 10, // 10MB
            },
            git: GitConfig {
                enable_analysis: true,
                enable_conflict_prediction: true,
                enable_fast_diff: true,
                max_repo_size_mb: 1000,
            },
            session: SessionConfig {
                auto_save: true,
                save_dir: "~/.claude-code/sessions".to_string(),
                max_sessions: 100,
                session_timeout_hours: 24,
            },
            performance: PerformanceConfig {
                enable_metrics: true,
                metrics_port: 9090,
                enable_tracing: true,
                log_level: "info".to_string(),
            },
        }
    }
}

impl ClaudeCodeConfig {
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
            "api.model" => self.api.model = value.to_string(),
            "api.base_url" => self.api.base_url = value.to_string(),
            "api.temperature" => {
                self.api.temperature = value.parse()
                    .context("Invalid float value")?;
            }
            "api.max_tokens" => {
                self.api.max_tokens = value.parse()
                    .context("Invalid integer value")?;
            }
            "hardware.enable_npu" => {
                self.hardware.enable_npu = value.parse()
                    .context("Invalid boolean value")?;
            }
            "hardware.enable_avx512" => {
                self.hardware.enable_avx512 = value.parse()
                    .context("Invalid boolean value")?;
            }
            "hardware.enable_gpu" => {
                self.hardware.enable_gpu = value.parse()
                    .context("Invalid boolean value")?;
            }
            "agents.max_concurrent" => {
                self.agents.max_concurrent = value.parse()
                    .context("Invalid integer value")?;
            }
            "git.enable_analysis" => {
                self.git.enable_analysis = value.parse()
                    .context("Invalid boolean value")?;
            }
            _ => anyhow::bail!("Unknown configuration key: {}", key),
        }

        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate API config
        if self.api.max_tokens == 0 {
            anyhow::bail!("max_tokens must be greater than 0");
        }

        if !(0.0..=2.0).contains(&self.api.temperature) {
            anyhow::bail!("temperature must be between 0.0 and 2.0");
        }

        // Validate hardware config
        if self.hardware.p_cores == 0 && self.hardware.e_cores == 0 {
            anyhow::bail!("Must have at least one core enabled");
        }

        // Validate agent config
        if self.agents.max_concurrent == 0 {
            anyhow::bail!("max_concurrent must be greater than 0");
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
        let config = ClaudeCodeConfig::default();
        assert_eq!(config.api.model, "claude-3-5-sonnet-20241022");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_save_and_load() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = ClaudeCodeConfig::default();

        config.save(temp_file.path()).unwrap();
        let loaded = ClaudeCodeConfig::load(temp_file.path()).unwrap();

        assert_eq!(config.api.model, loaded.api.model);
    }

    #[test]
    fn test_validation() {
        let mut config = ClaudeCodeConfig::default();
        assert!(config.validate().is_ok());

        config.api.max_tokens = 0;
        assert!(config.validate().is_err());
    }
}
