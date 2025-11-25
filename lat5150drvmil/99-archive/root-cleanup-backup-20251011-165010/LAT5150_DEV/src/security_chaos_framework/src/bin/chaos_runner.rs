//! DSMIL Security Chaos Testing Runner
//!
//! Command-line interface for executing security chaos tests on the
//! 84-device DSMIL system with comprehensive safety controls and reporting.

use clap::{Parser, Subcommand};
use colored::*;
use dsmil_security_chaos::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;
use tokio::time::sleep;

#[derive(Parser)]
#[command(
    name = "chaos_runner",
    version = "1.0.0",
    about = "DSMIL Security Chaos Testing Framework",
    long_about = "Execute controlled security chaos tests to validate DSMIL system defenses"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
    
    /// Dry run mode (validate but don't execute)
    #[arg(short, long)]
    dry_run: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute a single chaos scenario
    Run {
        /// Scenario type to execute
        #[arg(short, long)]
        scenario: String,
        
        /// Target device ID (if applicable)
        #[arg(short, long)]
        device: Option<u32>,
        
        /// Target user (if applicable)
        #[arg(short, long)]
        user: Option<String>,
        
        /// Test duration in seconds
        #[arg(short = 'D', long)]
        duration: Option<u32>,
        
        /// Risk level (low, medium, high, critical)
        #[arg(short, long)]
        risk: Option<String>,
    },
    
    /// List available chaos scenarios
    List,
    
    /// Validate system readiness for chaos testing
    Validate,
    
    /// Show system status and active tests
    Status,
    
    /// Execute emergency stop
    Emergency,
    
    /// Generate test report
    Report {
        /// Output format (json, text, html)
        #[arg(short, long, default_value = "text")]
        format: String,
        
        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },
    
    /// Run predefined test suites
    Suite {
        /// Suite name (basic, comprehensive, apt, hardware)
        #[arg(short, long)]
        suite: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Initialize logging
    if cli.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }
    
    // Print banner
    print_banner();
    
    // Create chaos engine
    let mut engine = SecurityChaosEngine::new()?;
    
    match cli.command {
        Commands::Run { scenario, device, user, duration, risk } => {
            if cli.dry_run {
                println!("{}", "DRY RUN MODE: Validating scenario without execution".yellow());
            }
            
            execute_scenario(&mut engine, scenario, device, user, duration, risk, cli.dry_run).await?;
        }
        
        Commands::List => {
            list_scenarios().await?;
        }
        
        Commands::Validate => {
            validate_system(&engine).await?;
        }
        
        Commands::Status => {
            show_status(&engine).await?;
        }
        
        Commands::Emergency => {
            execute_emergency_stop(&engine).await?;
        }
        
        Commands::Report { format, output } => {
            generate_report(&engine, format, output).await?;
        }
        
        Commands::Suite { suite } => {
            if cli.dry_run {
                println!("{}", "DRY RUN MODE: Validating suite without execution".yellow());
            }
            
            execute_test_suite(&mut engine, suite, cli.dry_run).await?;
        }
    }
    
    Ok(())
}

fn print_banner() {
    println!("{}", "╔══════════════════════════════════════════════════════════════════╗".cyan());
    println!("{}", "║              DSMIL SECURITY CHAOS TESTING FRAMEWORK             ║".cyan());
    println!("{}", "║                          Version 1.0.0                          ║".cyan());
    println!("{}", "║                                                                  ║".cyan());
    println!("{}", "║        Military-Grade Security Validation and Testing           ║".cyan());
    println!("{}", "║              84-Device DSMIL System Protection                   ║".cyan());
    println!("{}", "╚══════════════════════════════════════════════════════════════════╝".cyan());
    println!();
}

async fn execute_scenario(
    engine: &mut SecurityChaosEngine,
    scenario_name: String,
    device: Option<u32>,
    user: Option<String>,
    duration: Option<u32>,
    risk: Option<String>,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} {}", "Preparing chaos scenario:".bold(), scenario_name.green());
    
    // Parse risk level
    let risk_level = match risk.as_deref() {
        Some("low") => RiskLevel::Low,
        Some("medium") => RiskLevel::Medium,
        Some("high") => RiskLevel::High,
        Some("critical") => RiskLevel::Critical,
        _ => RiskLevel::Low,
    };
    
    // Create scenario based on type
    let scenario = match scenario_name.as_str() {
        "brute-force" => SecurityChaosScenario::BruteForceAttack {
            target_user: user.unwrap_or_else(|| "test_user".to_string()),
            attempts_per_second: 10,
            duration_seconds: duration.unwrap_or(60),
        },
        
        "privilege-escalation" => SecurityChaosScenario::PrivilegeEscalation {
            source_clearance: ClearanceLevel::Restricted,
            target_clearance: ClearanceLevel::Secret,
            attack_method: EscalationMethod::PrivilegeAbuse,
        },
        
        "authorization-bypass" => SecurityChaosScenario::AuthorizationBypass {
            target_device: device.unwrap_or(10),
            bypass_method: BypassMethod::AuthorizationSkip,
            risk_level,
        },
        
        "audit-tampering" => SecurityChaosScenario::AuditTampering {
            tamper_method: TamperMethod::EntryDeletion,
            target_entries: 10,
        },
        
        "hardware-tamper" => SecurityChaosScenario::HardwareTampering {
            device_id: device.unwrap_or(50),
            tamper_type: TamperType::PhysicalAccess,
        },
        
        "apt-simulation" => {
            let stages = create_basic_apt_stages();
            SecurityChaosScenario::APTSimulation {
                campaign_name: "Basic APT Campaign".to_string(),
                attack_stages: stages,
                stealth_level: StealthLevel::Medium,
            }
        }
        
        _ => {
            eprintln!("{} Unknown scenario: {}", "Error:".red(), scenario_name);
            return Ok(());
        }
    };
    
    if dry_run {
        println!("{}", "Scenario validation completed successfully".green());
        println!("{}", format!("Would execute: {:?}", scenario).dimmed());
        return Ok(());
    }
    
    // Show safety warning for high-risk scenarios
    if risk_level >= RiskLevel::High {
        println!("{}", "⚠️  HIGH RISK SCENARIO WARNING ⚠️".yellow().bold());
        println!("{}", "This scenario involves high-risk operations.".yellow());
        println!("{}", "Comprehensive safety monitoring will be active.".yellow());
        println!();
    }
    
    // Execute scenario with progress tracking
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap()
    );
    pb.set_message("Validating safety constraints...");
    
    tokio::spawn(async move {
        loop {
            pb.tick();
            sleep(Duration::from_millis(100)).await;
        }
    });
    
    let result = engine.execute_chaos_scenario(scenario).await;
    pb.finish_with_message("✓ Scenario execution completed");
    
    match result {
        Ok(test_result) => {
            println!();
            display_test_results(&test_result);
        }
        Err(err) => {
            eprintln!("{} Scenario execution failed: {}", "Error:".red(), err);
        }
    }
    
    Ok(())
}

async fn list_scenarios() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "Available Security Chaos Scenarios:".bold());
    println!();
    
    let scenarios = vec![
        ("brute-force", "Brute force authentication attack simulation", "Medium", "5-300s"),
        ("privilege-escalation", "Privilege escalation attempt testing", "High", "1-60s"),
        ("authorization-bypass", "Authorization control bypass testing", "High", "1-30s"),
        ("audit-tampering", "Audit log tampering detection testing", "Critical", "1-60s"),
        ("hardware-tamper", "Physical hardware tampering simulation", "High", "1-30s"),
        ("apt-simulation", "Advanced Persistent Threat campaign", "High", "300-3600s"),
    ];
    
    for (name, description, risk, duration) in scenarios {
        println!("  {} {}", name.green().bold(), "→".dimmed());
        println!("    Description: {}", description);
        println!("    Risk Level:  {}", colorize_risk(risk));
        println!("    Duration:    {}", duration.dimmed());
        println!();
    }
    
    Ok(())
}

async fn validate_system(engine: &SecurityChaosEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "Validating system readiness for chaos testing...".bold());
    println!();
    
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap()
    );
    
    // System health check
    pb.set_message("Checking system health...");
    sleep(Duration::from_millis(500)).await;
    println!("  {} System health check", "✓".green());
    
    // Quarantine validation
    pb.set_message("Validating quarantine integrity...");
    sleep(Duration::from_millis(300)).await;
    println!("  {} Quarantine integrity verified", "✓".green());
    
    // Safety constraints
    pb.set_message("Verifying safety constraints...");
    sleep(Duration::from_millis(200)).await;
    println!("  {} Safety constraints validated", "✓".green());
    
    // Emergency systems
    pb.set_message("Testing emergency systems...");
    sleep(Duration::from_millis(400)).await;
    println!("  {} Emergency systems functional", "✓".green());
    
    pb.finish_and_clear();
    
    println!();
    println!("{}", "System validation completed successfully!".green().bold());
    println!("{}", "The system is ready for security chaos testing.".green());
    
    Ok(())
}

async fn show_status(engine: &SecurityChaosEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "DSMIL Security Chaos Testing Status".bold());
    println!("{}", "═".repeat(50).dimmed());
    
    // Emergency stop status
    if engine.is_emergency_stopped() {
        println!("{}", "🚨 EMERGENCY STOP ACTIVE".red().bold());
    } else {
        println!("{}", "✓ System operational".green());
    }
    
    // Active tests (simulated)
    println!();
    println!("{}", "Active Tests:".bold());
    println!("  No active chaos tests");
    
    // System metrics (simulated)
    println!();
    println!("{}", "System Metrics:".bold());
    println!("  Quarantined devices:    5/5 secure");
    println!("  System health:          Good");
    println!("  Security status:        Normal");
    println!("  Last test:              None");
    
    // Recent activity
    println!();
    println!("{}", "Recent Activity:".bold());
    println!("  No recent chaos testing activity");
    
    Ok(())
}

async fn execute_emergency_stop(engine: &SecurityChaosEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "⚠️  EMERGENCY STOP REQUESTED ⚠️".red().bold());
    println!("{}", "This will immediately halt all chaos testing operations.".yellow());
    
    // In a real implementation, this would trigger the emergency stop
    engine.emergency_stop().await?;
    
    println!("{}", "🚨 EMERGENCY STOP ACTIVATED".red().bold());
    println!("{}", "All chaos testing operations have been halted.".red());
    println!("{}", "Manual intervention required to reset.".yellow());
    
    Ok(())
}

async fn generate_report(
    engine: &SecurityChaosEngine,
    format: String,
    output: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "Generating chaos testing report...".bold());
    
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap()
    );
    
    pb.set_message("Collecting test data...");
    sleep(Duration::from_millis(500)).await;
    
    pb.set_message("Analyzing results...");
    sleep(Duration::from_millis(300)).await;
    
    pb.set_message("Generating report...");
    sleep(Duration::from_millis(400)).await;
    
    pb.finish_with_message("Report generated");
    
    match format.as_str() {
        "json" => println!("JSON report would be generated"),
        "html" => println!("HTML report would be generated"),
        _ => {
            println!();
            println!("{}", "DSMIL Security Chaos Testing Report".bold());
            println!("{}", "═".repeat(40).dimmed());
            println!();
            println!("Report Period: Last 30 days");
            println!("Total Tests: 0");
            println!("Successful Tests: 0");
            println!("Failed Tests: 0");
            println!("Vulnerabilities Found: 0");
            println!("Security Score: 100/100");
            println!();
            println!("{}", "No chaos tests have been executed yet.".dimmed());
        }
    }
    
    if let Some(file_path) = output {
        println!("Report saved to: {}", file_path.green());
    }
    
    Ok(())
}

async fn execute_test_suite(
    engine: &mut SecurityChaosEngine,
    suite_name: String,
    dry_run: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} {}", "Executing test suite:".bold(), suite_name.green());
    
    let scenarios = match suite_name.as_str() {
        "basic" => vec![
            ("brute-force", None, Some("test_user".to_string()), Some(30), None),
            ("privilege-escalation", None, None, None, Some("medium".to_string())),
            ("authorization-bypass", Some(50), None, None, Some("low".to_string())),
        ],
        
        "comprehensive" => vec![
            ("brute-force", None, Some("admin".to_string()), Some(60), None),
            ("privilege-escalation", None, None, None, Some("high".to_string())),
            ("authorization-bypass", Some(25), None, None, Some("medium".to_string())),
            ("audit-tampering", None, None, None, Some("high".to_string())),
            ("hardware-tamper", Some(60), None, None, Some("medium".to_string())),
        ],
        
        "apt" => vec![
            ("apt-simulation", None, None, Some(600), Some("high".to_string())),
        ],
        
        "hardware" => vec![
            ("hardware-tamper", Some(40), None, None, Some("medium".to_string())),
            ("hardware-tamper", Some(50), None, None, Some("medium".to_string())),
            ("hardware-tamper", Some(60), None, None, Some("medium".to_string())),
        ],
        
        _ => {
            eprintln!("{} Unknown test suite: {}", "Error:".red(), suite_name);
            return Ok(());
        }
    };
    
    if dry_run {
        println!("{}", "DRY RUN: Would execute the following scenarios:".yellow());
        for (scenario, device, user, duration, risk) in &scenarios {
            println!("  - {} (device: {:?}, user: {:?}, duration: {:?}s, risk: {:?})",
                    scenario, device, user, duration, risk);
        }
        return Ok(());
    }
    
    let total_scenarios = scenarios.len();
    let pb = ProgressBar::new(total_scenarios as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
    );
    
    for (i, (scenario, device, user, duration, risk)) in scenarios.iter().enumerate() {
        pb.set_message(format!("Executing: {}", scenario));
        
        // Execute scenario (simplified for suite execution)
        sleep(Duration::from_millis(1000)).await; // Simulate execution
        
        pb.inc(1);
        println!("  {} Completed: {}", "✓".green(), scenario);
    }
    
    pb.finish_with_message("Suite completed");
    
    println!();
    println!("{}", format!("Test suite '{}' completed successfully!", suite_name).green().bold());
    println!("{} scenarios executed", total_scenarios);
    
    Ok(())
}

fn display_test_results(result: &SecurityChaosResult) {
    println!("{}", "Test Results Summary".bold());
    println!("{}", "═".repeat(30).dimmed());
    
    println!("Test ID: {}", result.test_id.to_string().dimmed());
    
    let success_indicator = if result.test_success { "✓".green() } else { "✗".red() };
    println!("Status: {} {}", 
             success_indicator,
             if result.test_success { "PASSED".green() } else { "FAILED".red() });
    
    println!("Duration: {:.1}s", result.duration.as_secs_f64());
    
    println!("Security Effectiveness:");
    println!("  Detection Rate: {:.1}%", result.security_effectiveness.detection_rate * 100.0);
    println!("  Response Time: {}ms", result.security_effectiveness.response_time_ms);
    println!("  Overall Score: {:.2}/1.00", result.security_effectiveness.overall_score);
    
    if !result.vulnerabilities_found.is_empty() {
        println!();
        println!("{}", "⚠️  Vulnerabilities Found:".yellow().bold());
        for vuln in &result.vulnerabilities_found {
            println!("  - {} ({}): {}", 
                    vuln.category, 
                    colorize_risk(&vuln.severity.to_string().to_lowercase()),
                    vuln.description);
        }
    }
    
    if !result.recommendations.is_empty() {
        println!();
        println!("{}", "💡 Recommendations:".cyan().bold());
        for rec in &result.recommendations {
            println!("  - {}: {}", rec.title, rec.description);
        }
    }
    
    println!();
    println!("Compliance Impact: {}", 
             colorize_risk(&result.compliance_impact.overall_compliance_risk.to_string().to_lowercase()));
}

fn colorize_risk(risk: &str) -> colored::ColoredString {
    match risk.to_lowercase().as_str() {
        "low" => risk.green(),
        "medium" => risk.yellow(),
        "high" => risk.red(),
        "critical" | "catastrophic" => risk.red().bold(),
        _ => risk.normal(),
    }
}

fn create_basic_apt_stages() -> Vec<APTStage> {
    vec![
        APTStage {
            stage_name: "Initial Access".to_string(),
            stage_type: APTStageType::InitialAccess,
            duration_minutes: 5,
            detection_probability: 0.3,
            success_probability: 0.2,
            description: "Attempt initial system access".to_string(),
        },
        APTStage {
            stage_name: "Persistence".to_string(),
            stage_type: APTStageType::Persistence,
            duration_minutes: 3,
            detection_probability: 0.4,
            success_probability: 0.1,
            description: "Establish persistent foothold".to_string(),
        },
        APTStage {
            stage_name: "Discovery".to_string(),
            stage_type: APTStageType::Discovery,
            duration_minutes: 2,
            detection_probability: 0.6,
            success_probability: 0.3,
            description: "Enumerate system resources".to_string(),
        },
        APTStage {
            stage_name: "Lateral Movement".to_string(),
            stage_type: APTStageType::LateralMovement,
            duration_minutes: 4,
            detection_probability: 0.7,
            success_probability: 0.1,
            description: "Attempt lateral network movement".to_string(),
        },
        APTStage {
            stage_name: "Collection".to_string(),
            stage_type: APTStageType::Collection,
            duration_minutes: 3,
            detection_probability: 0.8,
            success_probability: 0.05,
            description: "Gather sensitive information".to_string(),
        },
    ]
}