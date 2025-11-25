//! Claude Code - High-Performance Coding Agent Client
//!
//! Production-ready Claude Code client with improvements from claude-backups:
//! - NPU acceleration for inference
//! - AVX2/AVX-512 SIMD optimizations
//! - Ultra-low latency binary IPC
//! - Git intelligence (ShadowGit)
//! - Agent orchestration
//! - Hardware acceleration
//!
//! Based on Dell Latitude 5450 Covert Edition with Intel Core Ultra 7 (Meteor Lake)

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{debug, error, info, Level};

mod agent;
mod client;
mod config;
mod git_analysis;
mod ipc;
mod mcp;
mod session;
mod simd;

use client::ClaudeCodeClient;
use config::ClaudeCodeConfig;

#[derive(Parser, Debug)]
#[command(name = "claude-code")]
#[command(about = "High-performance Claude Code client with NPU acceleration", long_about = None)]
#[command(version)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Enable debug mode
    #[arg(short, long)]
    debug: bool,

    /// Enable NPU acceleration
    #[arg(long)]
    npu: bool,

    /// Enable AVX-512 (if available)
    #[arg(long)]
    avx512: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start interactive coding session
    Interactive {
        /// Initial prompt
        prompt: Option<String>,

        /// Project directory
        #[arg(short, long)]
        project: Option<PathBuf>,
    },

    /// Execute single task and exit
    Exec {
        /// Task to execute
        task: String,

        /// Output format (text, json, markdown)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Enable streaming
        #[arg(short, long)]
        stream: bool,
    },

    /// Start MCP server mode
    Mcp {
        /// Transport protocol (stdio, http, ipc)
        #[arg(short, long, default_value = "stdio")]
        transport: String,

        /// Port for HTTP transport
        #[arg(short, long)]
        port: Option<u16>,

        /// Shared memory name for IPC
        #[arg(long)]
        shm_name: Option<String>,
    },

    /// Agent orchestration commands
    Agent {
        #[command(subcommand)]
        action: AgentCommands,
    },

    /// Git analysis commands (ShadowGit)
    Git {
        #[command(subcommand)]
        action: GitCommands,
    },

    /// Session management
    Session {
        #[command(subcommand)]
        action: SessionCommands,
    },

    /// Benchmark performance
    Bench {
        /// Benchmark suite (all, ipc, simd, git, agent)
        #[arg(short, long, default_value = "all")]
        suite: String,

        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: usize,
    },

    /// Configuration management
    Config {
        #[command(subcommand)]
        action: ConfigCommands,
    },
}

#[derive(Subcommand, Debug)]
enum AgentCommands {
    /// List available agents
    List,

    /// Show agent details
    Info {
        /// Agent ID
        agent_id: String,
    },

    /// Execute task with specific agent
    Execute {
        /// Agent ID
        agent_id: String,

        /// Task description
        task: String,
    },

    /// Agent performance statistics
    Stats,
}

#[derive(Subcommand, Debug)]
enum GitCommands {
    /// Analyze repository
    Analyze {
        /// Repository path
        #[arg(short, long)]
        repo: Option<PathBuf>,
    },

    /// Predict merge conflicts
    Conflicts {
        /// Base branch
        base: String,

        /// Compare branch
        compare: String,
    },

    /// Fast diff with SIMD
    Diff {
        /// Commit/branch A
        a: String,

        /// Commit/branch B
        b: String,
    },

    /// Repository intelligence
    Intelligence {
        /// Repository path
        #[arg(short, long)]
        repo: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum SessionCommands {
    /// Create new session
    New {
        /// Session name
        name: Option<String>,
    },

    /// List sessions
    List,

    /// Resume session
    Resume {
        /// Session ID or name
        session: String,
    },

    /// Delete session
    Delete {
        /// Session ID or name
        session: String,
    },
}

#[derive(Subcommand, Debug)]
enum ConfigCommands {
    /// Show configuration
    Show,

    /// Set configuration value
    Set {
        key: String,
        value: String,
    },

    /// Initialize default configuration
    Init,

    /// Validate configuration
    Validate,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.debug {
        Level::DEBUG
    } else if cli.verbose {
        Level::INFO
    } else {
        Level::WARN
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .with_thread_ids(true)
        .init();

    info!("Claude Code v{}", env!("CARGO_PKG_VERSION"));
    info!("Meteor Lake optimizations: AVX2 + FMA + AES");

    // Load configuration
    let config_path = cli.config.unwrap_or_else(|| {
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".claude-code")
            .join("config.toml")
    });

    debug!("Loading configuration from: {:?}", config_path);

    let mut config = if config_path.exists() {
        ClaudeCodeConfig::load(&config_path)?
    } else {
        info!("Configuration not found, using defaults");
        ClaudeCodeConfig::default()
    };

    // Apply CLI overrides
    if cli.npu {
        config.hardware.enable_npu = true;
    }
    if cli.avx512 {
        config.hardware.enable_avx512 = true;
    }

    // Execute command
    match cli.command {
        Some(Commands::Interactive { prompt, project }) => {
            run_interactive(&config, prompt.as_deref(), project.as_deref()).await?;
        }

        Some(Commands::Exec { task, format, stream }) => {
            run_exec(&config, &task, &format, stream).await?;
        }

        Some(Commands::Mcp { transport, port, shm_name }) => {
            run_mcp_server(&config, &transport, port, shm_name.as_deref()).await?;
        }

        Some(Commands::Agent { action }) => {
            handle_agent_commands(&config, action).await?;
        }

        Some(Commands::Git { action }) => {
            handle_git_commands(&config, action).await?;
        }

        Some(Commands::Session { action }) => {
            handle_session_commands(&config, action).await?;
        }

        Some(Commands::Bench { suite, iterations }) => {
            run_benchmarks(&config, &suite, iterations).await?;
        }

        Some(Commands::Config { action }) => {
            handle_config_commands(&config_path, action).await?;
        }

        None => {
            // Default to interactive mode
            run_interactive(&config, None, None).await?;
        }
    }

    Ok(())
}

async fn run_interactive(
    config: &ClaudeCodeConfig,
    initial_prompt: Option<&str>,
    project_dir: Option<&Path>,
) -> Result<()> {
    info!("Starting interactive coding session");

    let client = ClaudeCodeClient::new(config).await?;

    // Display welcome banner
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Claude Code - High-Performance Coding Agent             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Hardware Acceleration:");
    if config.hardware.enable_npu {
        println!("  ✓ NPU (Intel AI Boost)");
    }
    if config.hardware.enable_avx512 {
        println!("  ✓ AVX-512 SIMD");
    } else {
        println!("  ✓ AVX2 SIMD");
    }
    if config.hardware.enable_gpu {
        println!("  ✓ GPU (OpenVINO)");
    }
    println!();
    println!("Features:");
    println!("  ✓ Git Intelligence (ShadowGit)");
    println!("  ✓ Agent Orchestration");
    println!("  ✓ Binary IPC (50ns-10µs latency)");
    println!("  ✓ Ultra-fast diff processing");
    println!();
    println!("Commands: /help, /agents, /git, /session, /exit");
    println!();

    // Handle initial prompt
    if let Some(prompt) = initial_prompt {
        println!("User: {}", prompt);
        execute_prompt(&client, prompt, project_dir).await?;
    }

    // Interactive loop
    loop {
        print!("\n> ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "/exit" || input == "/quit" {
            println!("Goodbye!");
            break;
        }

        if input.starts_with('/') {
            handle_slash_command(&client, input).await?;
            continue;
        }

        execute_prompt(&client, input, project_dir).await?;
    }

    Ok(())
}

async fn execute_prompt(
    client: &ClaudeCodeClient,
    prompt: &str,
    _project_dir: Option<&Path>,
) -> Result<()> {
    use std::io::{self, Write};

    print!("\nClaude: ");
    io::stdout().flush()?;

    let start = std::time::Instant::now();

    // Stream response
    let mut stream = client.execute_streaming(prompt).await?;
    let mut char_count = 0;

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(text) => {
                print!("{}", text);
                io::stdout().flush()?;
                char_count += text.len();
            }
            Err(e) => {
                error!("Streaming error: {}", e);
                break;
            }
        }
    }

    let duration = start.elapsed();
    println!();
    debug!(
        "Response: {} chars in {:.2}s ({:.0} chars/s)",
        char_count,
        duration.as_secs_f64(),
        char_count as f64 / duration.as_secs_f64()
    );

    Ok(())
}

async fn handle_slash_command(client: &ClaudeCodeClient, command: &str) -> Result<()> {
    match command {
        "/help" => {
            println!("\nAvailable commands:");
            println!("  /help       - Show this help");
            println!("  /agents     - List available agents");
            println!("  /git        - Git intelligence");
            println!("  /session    - Session management");
            println!("  /stats      - Performance statistics");
            println!("  /exit       - Exit interactive mode");
        }

        "/agents" => {
            let agents = client.list_agents().await?;
            println!("\nAvailable agents ({}):", agents.len());
            for agent in agents {
                println!("  - {} ({})", agent.name, agent.capabilities.join(", "));
            }
        }

        "/stats" => {
            let stats = client.get_stats().await?;
            println!("\nPerformance Statistics:");
            println!("  Messages: {}", stats.message_count);
            println!("  Avg latency: {:.2}ms", stats.avg_latency_ms);
            println!("  Total tokens: {}", stats.total_tokens);
        }

        _ => {
            println!("\nUnknown command: {}", command);
            println!("Type '/help' for available commands");
        }
    }

    Ok(())
}

async fn run_exec(
    config: &ClaudeCodeConfig,
    task: &str,
    format: &str,
    stream: bool,
) -> Result<()> {
    let client = ClaudeCodeClient::new(config).await?;

    if stream {
        let mut response_stream = client.execute_streaming(task).await?;
        while let Some(chunk) = response_stream.next().await {
            match chunk {
                Ok(text) => print!("{}", text),
                Err(e) => {
                    error!("Stream error: {}", e);
                    break;
                }
            }
        }
        println!();
    } else {
        let response = client.execute(task).await?;

        match format {
            "json" => {
                println!("{}", serde_json::to_string_pretty(&response)?);
            }
            "markdown" => {
                println!("# Response\n\n{}", response.content);
            }
            _ => {
                println!("{}", response.content);
            }
        }
    }

    Ok(())
}

async fn run_mcp_server(
    config: &ClaudeCodeConfig,
    transport: &str,
    port: Option<u16>,
    shm_name: Option<&str>,
) -> Result<()> {
    info!("Starting MCP server: transport={}", transport);

    match transport {
        "stdio" => {
            mcp::run_stdio_server(config).await?;
        }

        "http" => {
            let port = port.unwrap_or(6282);
            mcp::run_http_server(config, port).await?;
        }

        "ipc" => {
            let name = shm_name.unwrap_or("claude-code-ipc");
            mcp::run_ipc_server(config, name).await?;
        }

        _ => {
            anyhow::bail!("Unsupported transport: {}", transport);
        }
    }

    Ok(())
}

async fn handle_agent_commands(config: &ClaudeCodeConfig, action: AgentCommands) -> Result<()> {
    use agent::AgentOrchestrator;

    let orchestrator = AgentOrchestrator::new(config)?;

    match action {
        AgentCommands::List => {
            let agents = orchestrator.list_agents();
            println!("\nAvailable agents ({}):", agents.len());
            for agent in agents {
                println!("  - {}: {}", agent.id, agent.description);
            }
        }

        AgentCommands::Info { agent_id } => {
            if let Some(agent) = orchestrator.get_agent(&agent_id) {
                println!("\nAgent: {}", agent.name);
                println!("ID: {}", agent.id);
                println!("Capabilities: {}", agent.capabilities.join(", "));
                println!("Hardware: {}", agent.preferred_hardware);
            } else {
                println!("Agent not found: {}", agent_id);
            }
        }

        AgentCommands::Execute { agent_id, task } => {
            println!("Executing task with agent {}...", agent_id);
            let result = orchestrator.execute_with_agent(&agent_id, &task).await?;
            println!("\nResult:\n{}", result);
        }

        AgentCommands::Stats => {
            let stats = orchestrator.get_stats();
            println!("\nAgent Orchestration Statistics:");
            println!("  Total agents: {}", stats.total_agents);
            println!("  Tasks executed: {}", stats.tasks_executed);
            println!("  Avg latency: {:.2}ms", stats.avg_latency_ms);
        }
    }

    Ok(())
}

async fn handle_git_commands(_config: &ClaudeCodeConfig, action: GitCommands) -> Result<()> {
    use git_analysis::ShadowGit;

    match action {
        GitCommands::Analyze { repo } => {
            let repo_path = repo.unwrap_or_else(|| PathBuf::from("."));
            println!("Analyzing repository: {:?}", repo_path);

            let shadow = ShadowGit::new(&repo_path)?;
            let analysis = shadow.analyze().await?;

            println!("\nRepository Analysis:");
            println!("  Commits: {}", analysis.commit_count);
            println!("  Branches: {}", analysis.branch_count);
            println!("  Contributors: {}", analysis.contributor_count);
            println!("  Complexity score: {:.2}", analysis.complexity_score);
        }

        GitCommands::Conflicts { base, compare } => {
            let shadow = ShadowGit::new(".")?;
            let conflicts = shadow.predict_conflicts(&base, &compare).await?;

            println!("\nPredicted conflicts: {}", conflicts.len());
            for conflict in conflicts {
                println!("  - {}: {}", conflict.file, conflict.reason);
            }
        }

        GitCommands::Diff { a, b } => {
            let shadow = ShadowGit::new(".")?;
            let start = std::time::Instant::now();
            let diff = shadow.fast_diff(&a, &b).await?;
            let duration = start.elapsed();

            println!("\nDiff computed in {:.2}ms (SIMD accelerated)", duration.as_secs_f64() * 1000.0);
            println!("Changes: {} files, +{} -{}", diff.files_changed, diff.insertions, diff.deletions);
        }

        GitCommands::Intelligence { repo } => {
            let repo_path = repo.unwrap_or_else(|| PathBuf::from("."));
            let shadow = ShadowGit::new(&repo_path)?;
            let intel = shadow.intelligence().await?;

            println!("\nRepository Intelligence:");
            println!("{}", serde_json::to_string_pretty(&intel)?);
        }
    }

    Ok(())
}

async fn handle_session_commands(_config: &ClaudeCodeConfig, action: SessionCommands) -> Result<()> {
    use session::SessionManager;

    let manager = SessionManager::new()?;

    match action {
        SessionCommands::New { name } => {
            let session = manager.create_session(name.as_deref()).await?;
            println!("Created session: {} (ID: {})", session.name, session.id);
        }

        SessionCommands::List => {
            let sessions = manager.list_sessions().await?;
            println!("\nSessions ({}):", sessions.len());
            for session in sessions {
                println!("  - {} (ID: {}, created: {})", session.name, session.id, session.created_at);
            }
        }

        SessionCommands::Resume { session } => {
            let s = manager.load_session(&session).await?;
            println!("Resumed session: {} ({} messages)", s.name, s.message_count);
        }

        SessionCommands::Delete { session } => {
            manager.delete_session(&session).await?;
            println!("Deleted session: {}", session);
        }
    }

    Ok(())
}

async fn run_benchmarks(config: &ClaudeCodeConfig, suite: &str, iterations: usize) -> Result<()> {
    println!("Running benchmarks: suite={}, iterations={}", suite, iterations);

    match suite {
        "ipc" => {
            ipc::benchmark_ipc(iterations).await?;
        }

        "simd" => {
            simd::benchmark_simd(iterations).await?;
        }

        "git" => {
            git_analysis::benchmark_git(iterations).await?;
        }

        "agent" => {
            agent::benchmark_agents(config, iterations).await?;
        }

        "all" => {
            ipc::benchmark_ipc(iterations).await?;
            simd::benchmark_simd(iterations).await?;
            git_analysis::benchmark_git(iterations).await?;
            agent::benchmark_agents(config, iterations).await?;
        }

        _ => {
            anyhow::bail!("Unknown benchmark suite: {}", suite);
        }
    }

    Ok(())
}

async fn handle_config_commands(config_path: &Path, action: ConfigCommands) -> Result<()> {
    match action {
        ConfigCommands::Show => {
            let config = ClaudeCodeConfig::load(config_path)?;
            println!("\n{:#?}", config);
        }

        ConfigCommands::Set { key, value } => {
            let mut config = if config_path.exists() {
                ClaudeCodeConfig::load(config_path)?
            } else {
                ClaudeCodeConfig::default()
            };

            config.set_value(&key, &value)?;
            config.save(config_path)?;

            println!("✓ Configuration updated: {} = {}", key, value);
        }

        ConfigCommands::Init => {
            let config = ClaudeCodeConfig::default();

            if let Some(parent) = config_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            config.save(config_path)?;
            println!("✓ Configuration initialized at: {:?}", config_path);
        }

        ConfigCommands::Validate => {
            let config = ClaudeCodeConfig::load(config_path)?;
            config.validate()?;
            println!("✓ Configuration is valid");
        }
    }

    Ok(())
}
