//! Authentication management for Codex CLI

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{debug, info};

use crate::config::CodexConfig;

/// Login with ChatGPT account
pub async fn login_chatgpt<P: AsRef<Path>>(config_path: P) -> Result<()> {
    info!("Initiating ChatGPT login flow");

    // In a real implementation, this would:
    // 1. Open browser to ChatGPT auth page
    // 2. Wait for callback with session token
    // 3. Store token in config

    // For now, simulate the flow
    println!("\nChatGPT Authentication:");
    println!("1. Visit: https://chatgpt.com/auth");
    println!("2. Log in with your ChatGPT account");
    println!("3. Copy the session token from the response");
    println!("\nPaste your session token here:");

    let mut token = String::new();
    std::io::stdin().read_line(&mut token)?;
    let token = token.trim();

    if token.is_empty() {
        anyhow::bail!("No token provided");
    }

    // Load or create config
    let mut config = if config_path.as_ref().exists() {
        CodexConfig::load(&config_path)?
    } else {
        CodexConfig::default()
    };

    // Update authentication
    config.auth_method = "chatgpt".to_string();
    config.session_token = Some(token.to_string());

    // Save config
    config.save(&config_path)?;

    info!("Successfully authenticated with ChatGPT");

    Ok(())
}

/// Set OpenAI API key
pub async fn set_api_key<P: AsRef<Path>>(config_path: P, api_key: &str) -> Result<()> {
    info!("Setting OpenAI API key");

    // Validate API key format
    if !api_key.starts_with("sk-") {
        anyhow::bail!("Invalid API key format (should start with 'sk-')");
    }

    // Load or create config
    let mut config = if config_path.as_ref().exists() {
        CodexConfig::load(&config_path)?
    } else {
        CodexConfig::default()
    };

    // Update authentication
    config.auth_method = "api_key".to_string();
    config.api_key = Some(api_key.to_string());

    // Save config
    config.save(&config_path)?;

    info!("Successfully configured API key");

    Ok(())
}

/// Logout and clear credentials
pub async fn logout<P: AsRef<Path>>(config_path: P) -> Result<()> {
    info!("Logging out");

    if !config_path.as_ref().exists() {
        return Ok(());
    }

    // Load config
    let mut config = CodexConfig::load(&config_path)?;

    // Clear credentials
    config.session_token = None;
    config.api_key = None;

    // Save config
    config.save(&config_path)?;

    info!("Successfully logged out");

    Ok(())
}

/// Verify authentication status
pub async fn verify_auth(config: &CodexConfig) -> Result<bool> {
    debug!("Verifying authentication status");

    if !config.is_authenticated() {
        return Ok(false);
    }

    // In a real implementation, this would:
    // 1. Make a test API call
    // 2. Verify the credentials are valid
    // 3. Return true if valid, false otherwise

    // For now, just check if credentials exist
    Ok(config.is_authenticated())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_set_api_key() {
        let temp_file = NamedTempFile::new().unwrap();

        set_api_key(temp_file.path(), "sk-test123").await.unwrap();

        let config = CodexConfig::load(temp_file.path()).unwrap();
        assert_eq!(config.auth_method, "api_key");
        assert_eq!(config.api_key, Some("sk-test123".to_string()));
        assert!(config.is_authenticated());
    }

    #[tokio::test]
    async fn test_logout() {
        let temp_file = NamedTempFile::new().unwrap();

        // Set up authentication
        set_api_key(temp_file.path(), "sk-test123").await.unwrap();

        // Logout
        logout(temp_file.path()).await.unwrap();

        // Verify credentials cleared
        let config = CodexConfig::load(temp_file.path()).unwrap();
        assert!(config.api_key.is_none());
        assert!(!config.is_authenticated());
    }

    #[tokio::test]
    async fn test_verify_auth() {
        let mut config = CodexConfig::default();
        assert!(!verify_auth(&config).await.unwrap());

        config.api_key = Some("sk-test".to_string());
        config.auth_method = "api_key".to_string();
        assert!(verify_auth(&config).await.unwrap());
    }
}
