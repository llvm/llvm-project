//! Codex CLI - Local-first OpenAI Codex client
//!
//! Provides comprehensive developer support for codex CLI as a sub-agent
//! integrated with the LAT5150DRVMIL AI platform.
//!
//! Features:
//! - ChatGPT account authentication
//! - OpenAI API key support
//! - Streaming responses
//! - MCP (Model Context Protocol) integration
//! - Local configuration management
//! - Extended thinking/reasoning capabilities

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{debug, error, info, warn};

mod client;
mod config;
mod auth;
mod streaming;
mod mcp;

use client::CodexClient;
use config::CodexConfig;

#[derive(Parser, Debug)]
#[command(name = "codex")]
#[command(about = "Local-first OpenAI Codex CLI client", long_about = None)]
#[command(version)]
struct Cli {
    /// Configuration file path (defaults to ~/.codex/config.toml)
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Model to use (gpt-5-codex, gpt-5-codex-mini, etc.)
    #[arg(short, long)]
    model: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Interactive terminal agent
    Interactive {
        /// Initial prompt
        prompt: Option<String>,
    },

    /// Execute single command and exit
    Exec {
        /// Command to execute
        prompt: String,

        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Authentication management
    Auth {
        #[command(subcommand)]
        action: AuthCommands,
    },

    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigCommands,
    },

    /// Start MCP server mode
    Mcp {
        /// Transport protocol (stdio, http)
        #[arg(short, long, default_value = "stdio")]
        transport: String,

        /// Port for HTTP transport
        #[arg(short, long)]
        port: Option<u16>,
    },
}

#[derive(Subcommand, Debug)]
enum AuthCommands {
    /// Login with ChatGPT account
    Login,

    /// Login with API key
    ApiKey {
        /// OpenAI API key
        key: String,
    },

    /// Show current authentication status
    Status,

    /// Logout
    Logout,
}

#[derive(Subcommand, Debug)]
enum ConfigCommands {
    /// Show current configuration
    Show,

    /// Set configuration value
    Set {
        /// Configuration key
        key: String,

        /// Configuration value
        value: String,
    },

    /// Initialize default configuration
    Init,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level))
        )
        .init();

    info!("Codex CLI v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config_path = cli.config.unwrap_or_else(|| {
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".codex")
            .join("config.toml")
    });

    debug!("Loading configuration from: {:?}", config_path);

    let mut config = if config_path.exists() {
        CodexConfig::load(&config_path)?
    } else {
        warn!("Configuration not found, using defaults");
        CodexConfig::default()
    };

    // Override model if specified
    if let Some(model) = cli.model {
        config.model = model;
    }

    // Execute command
    match cli.command {
        Some(Commands::Interactive { prompt }) => {
            run_interactive(&config, prompt.as_deref()).await?;
        }

        Some(Commands::Exec { prompt, format }) => {
            run_exec(&config, &prompt, &format).await?;
        }

        Some(Commands::Auth { action }) => {
            handle_auth(&config_path, action).await?;
        }

        Some(Commands::Config { action }) => {
            handle_config(&config_path, action).await?;
        }

        Some(Commands::Mcp { transport, port }) => {
            run_mcp_server(&config, &transport, port).await?;
        }

        None => {
            // Default to interactive mode
            run_interactive(&config, None).await?;
        }
    }

    Ok(())
}

async fn run_interactive(config: &CodexConfig, initial_prompt: Option<&str>) -> Result<()> {
    info!("Starting interactive mode");

    let client = CodexClient::new(config)?;

    // Display welcome message
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Codex CLI - OpenAI Coding Agent (Local-First)       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Model: {}", config.model);
    println!("Type 'exit' or 'quit' to exit");
    println!("Type '/help' for available commands");
    println!();

    // Handle initial prompt if provided
    if let Some(prompt) = initial_prompt {
        println!("User: {}", prompt);
        execute_prompt(&client, prompt).await?;
    }

    // Interactive loop
    loop {
        print!("\nUser: ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "exit" || input == "quit" {
            println!("Goodbye!");
            break;
        }

        if input.starts_with('/') {
            handle_slash_command(&client, input).await?;
            continue;
        }

        execute_prompt(&client, input).await?;
    }

    Ok(())
}

async fn run_exec(config: &CodexConfig, prompt: &str, format: &str) -> Result<()> {
    debug!("Executing single command: {}", prompt);

    let client = CodexClient::new(config)?;
    let response = client.execute(prompt).await?;

    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        _ => {
            println!("{}", response.content);
        }
    }

    Ok(())
}

async fn execute_prompt(client: &CodexClient, prompt: &str) -> Result<()> {
    print!("\nCodex: ");
    use std::io::{self, Write};
    io::stdout().flush()?;

    let mut response_text = String::new();

    // Stream response
    let mut stream = client.execute_streaming(prompt).await?;

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => {
                print!("{}", text);
                io::stdout().flush()?;
                response_text.push_str(&text);
            }
            Err(e) => {
                error!("Streaming error: {}", e);
                break;
            }
        }
    }

    println!(); // Newline after response

    Ok(())
}

async fn handle_slash_command(client: &CodexClient, command: &str) -> Result<()> {
    match command {
        "/help" => {
            println!("\nAvailable commands:");
            println!("  /help          - Show this help message");
            println!("  /model [name]  - Show or change model");
            println!("  /clear         - Clear conversation history");
            println!("  /save [file]   - Save conversation to file");
            println!("  /load [file]   - Load conversation from file");
            println!("  /status        - Show client status");
            println!("  exit, quit     - Exit interactive mode");
        }

        "/status" => {
            let status = client.status().await?;
            println!("\nClient Status:");
            println!("  Model: {}", status.model);
            println!("  Authenticated: {}", status.authenticated);
            println!("  Messages: {}", status.message_count);
        }

        "/model" => {
            let status = client.status().await?;
            println!("\nCurrent model: {}", status.model);
        }

        cmd if cmd.starts_with("/model ") => {
            let new_model = cmd.strip_prefix("/model ").unwrap().trim();
            println!("Model switching not yet implemented: {}", new_model);
        }

        "/clear" => {
            client.clear_history().await?;
            println!("\nConversation history cleared");
        }

        _ => {
            println!("\nUnknown command: {}", command);
            println!("Type '/help' for available commands");
        }
    }

    Ok(())
}

async fn handle_auth(config_path: &PathBuf, action: AuthCommands) -> Result<()> {
    match action {
        AuthCommands::Login => {
            println!("ChatGPT account login:");
            println!("Opening browser for authentication...");
            auth::login_chatgpt(config_path).await?;
            println!("✓ Successfully authenticated");
        }

        AuthCommands::ApiKey { key } => {
            println!("Setting OpenAI API key...");
            auth::set_api_key(config_path, &key).await?;
            println!("✓ API key configured");
        }

        AuthCommands::Status => {
            let config = CodexConfig::load(config_path)?;
            println!("\nAuthentication Status:");
            println!("  Method: {}", config.auth_method);
            println!("  Configured: {}", config.is_authenticated());
        }

        AuthCommands::Logout => {
            auth::logout(config_path).await?;
            println!("✓ Logged out successfully");
        }
    }

    Ok(())
}

async fn handle_config(config_path: &PathBuf, action: ConfigCommands) -> Result<()> {
    match action {
        ConfigCommands::Show => {
            let config = CodexConfig::load(config_path)?;
            println!("\nCurrent Configuration:");
            println!("{:#?}", config);
        }

        ConfigCommands::Set { key, value } => {
            let mut config = if config_path.exists() {
                CodexConfig::load(config_path)?
            } else {
                CodexConfig::default()
            };

            config.set_value(&key, &value)?;
            config.save(config_path)?;

            println!("✓ Configuration updated: {} = {}", key, value);
        }

        ConfigCommands::Init => {
            let config = CodexConfig::default();

            // Create directory if it doesn't exist
            if let Some(parent) = config_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            config.save(config_path)?;
            println!("✓ Configuration initialized at: {:?}", config_path);
        }
    }

    Ok(())
}

async fn run_mcp_server(config: &CodexConfig, transport: &str, port: Option<u16>) -> Result<()> {
    info!("Starting MCP server mode: transport={}", transport);

    match transport {
        "stdio" => {
            mcp::run_stdio_server(config).await?;
        }

        "http" => {
            let port = port.unwrap_or(6281);
            mcp::run_http_server(config, port).await?;
        }

        _ => {
            anyhow::bail!("Unsupported transport: {}", transport);
        }
    }

    Ok(())
}
