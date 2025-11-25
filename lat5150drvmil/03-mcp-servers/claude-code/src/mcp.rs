//! MCP server implementation

use anyhow::Result;

use crate::config::ClaudeCodeConfig;

pub async fn run_stdio_server(_config: &ClaudeCodeConfig) -> Result<()> {
    println!("MCP stdio server running...");
    // Stub implementation
    Ok(())
}

pub async fn run_http_server(_config: &ClaudeCodeConfig, port: u16) -> Result<()> {
    println!("MCP HTTP server running on port {}...", port);
    // Stub implementation
    Ok(())
}

pub async fn run_ipc_server(_config: &ClaudeCodeConfig, name: &str) -> Result<()> {
    println!("MCP IPC server running (shm: {})...", name);
    // Stub implementation
    Ok(())
}
