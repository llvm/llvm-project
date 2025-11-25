//! Model Context Protocol (MCP) server implementation

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{self, BufRead, Write};
use tracing::{debug, error, info};

use crate::client::CodexClient;
use crate::config::CodexConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct McpRequest {
    jsonrpc: String,
    id: serde_json::Value,
    method: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct McpResponse {
    jsonrpc: String,
    id: serde_json::Value,

    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<McpError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct McpError {
    code: i32,
    message: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
}

/// Run MCP server in stdio mode
pub async fn run_stdio_server(config: &CodexConfig) -> Result<()> {
    info!("Starting MCP stdio server");

    let client = CodexClient::new(config)?;
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;

        debug!("Received request: {}", line);

        let request: McpRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                error!("Failed to parse request: {}", e);
                continue;
            }
        };

        let response = handle_mcp_request(&client, request).await;

        let response_json = serde_json::to_string(&response)?;
        debug!("Sending response: {}", response_json);

        writeln!(stdout, "{}", response_json)?;
        stdout.flush()?;
    }

    Ok(())
}

/// Run MCP server in HTTP mode
pub async fn run_http_server(config: &CodexConfig, port: u16) -> Result<()> {
    info!("Starting MCP HTTP server on port {}", port);

    // In a real implementation, this would use axum or actix-web
    // For now, just a placeholder

    println!("MCP HTTP server would run on port {}", port);
    println!("Press Ctrl+C to stop");

    // Keep server running
    tokio::signal::ctrl_c().await?;

    Ok(())
}

async fn handle_mcp_request(client: &CodexClient, request: McpRequest) -> McpResponse {
    let result = match request.method.as_str() {
        "initialize" => handle_initialize(&request),
        "tools/list" => handle_tools_list(),
        "tools/call" => handle_tool_call(client, &request).await,
        "resources/list" => handle_resources_list(),
        "prompts/list" => handle_prompts_list(),
        _ => {
            return McpResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(McpError {
                    code: -32601,
                    message: format!("Method not found: {}", request.method),
                    data: None,
                }),
            };
        }
    };

    match result {
        Ok(result) => McpResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: Some(result),
            error: None,
        },
        Err(e) => McpResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: None,
            error: Some(McpError {
                code: -32603,
                message: e.to_string(),
                data: None,
            }),
        },
    }
}

fn handle_initialize(_request: &McpRequest) -> Result<serde_json::Value> {
    debug!("Handling initialize request");

    Ok(json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {
                "listChanged": false
            },
            "resources": {
                "subscribe": false,
                "listChanged": false
            },
            "prompts": {
                "listChanged": false
            }
        },
        "serverInfo": {
            "name": "codex-cli",
            "version": env!("CARGO_PKG_VERSION")
        }
    }))
}

fn handle_tools_list() -> Result<serde_json::Value> {
    debug!("Handling tools/list request");

    Ok(json!({
        "tools": [
            {
                "name": "code_generation",
                "description": "Generate code based on natural language description",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Description of code to generate"
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language (optional)"
                        }
                    },
                    "required": ["description"]
                }
            },
            {
                "name": "code_review",
                "description": "Review code for issues and improvements",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to review"
                        },
                        "focus": {
                            "type": "string",
                            "description": "Review focus (security, performance, style, etc.)"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "debugging",
                "description": "Help debug code issues",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code with issues"
                        },
                        "error": {
                            "type": "string",
                            "description": "Error message or description"
                        }
                    },
                    "required": ["code", "error"]
                }
            },
            {
                "name": "refactoring",
                "description": "Refactor code for better quality",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to refactor"
                        },
                        "goal": {
                            "type": "string",
                            "description": "Refactoring goal (readability, performance, etc.)"
                        }
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "documentation",
                "description": "Generate documentation for code",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to document"
                        },
                        "format": {
                            "type": "string",
                            "description": "Documentation format (markdown, rst, etc.)"
                        }
                    },
                    "required": ["code"]
                }
            }
        ]
    }))
}

async fn handle_tool_call(client: &CodexClient, request: &McpRequest) -> Result<serde_json::Value> {
    debug!("Handling tools/call request");

    let params = request.params.as_ref()
        .context("Missing parameters")?;

    let tool_name = params.get("name")
        .and_then(|v| v.as_str())
        .context("Missing tool name")?;

    let tool_args = params.get("arguments")
        .context("Missing tool arguments")?;

    let prompt = build_tool_prompt(tool_name, tool_args)?;

    let response = client.execute(&prompt).await?;

    Ok(json!({
        "content": [
            {
                "type": "text",
                "text": response.content
            }
        ],
        "isError": false
    }))
}

fn build_tool_prompt(tool_name: &str, args: &serde_json::Value) -> Result<String> {
    let prompt = match tool_name {
        "code_generation" => {
            let description = args.get("description")
                .and_then(|v| v.as_str())
                .context("Missing description")?;

            let language = args.get("language")
                .and_then(|v| v.as_str())
                .unwrap_or("auto-detect");

            format!(
                "Generate {} code for: {}\n\nProvide clean, well-commented code.",
                language, description
            )
        }

        "code_review" => {
            let code = args.get("code")
                .and_then(|v| v.as_str())
                .context("Missing code")?;

            let focus = args.get("focus")
                .and_then(|v| v.as_str())
                .unwrap_or("general");

            format!(
                "Review this code with focus on {}:\n\n```\n{}\n```\n\nProvide specific feedback.",
                focus, code
            )
        }

        "debugging" => {
            let code = args.get("code")
                .and_then(|v| v.as_str())
                .context("Missing code")?;

            let error = args.get("error")
                .and_then(|v| v.as_str())
                .context("Missing error")?;

            format!(
                "Debug this code:\n\n```\n{}\n```\n\nError: {}\n\nExplain the issue and provide a fix.",
                code, error
            )
        }

        "refactoring" => {
            let code = args.get("code")
                .and_then(|v| v.as_str())
                .context("Missing code")?;

            let goal = args.get("goal")
                .and_then(|v| v.as_str())
                .unwrap_or("general improvement");

            format!(
                "Refactor this code for {}:\n\n```\n{}\n```\n\nProvide the refactored version.",
                goal, code
            )
        }

        "documentation" => {
            let code = args.get("code")
                .and_then(|v| v.as_str())
                .context("Missing code")?;

            let format = args.get("format")
                .and_then(|v| v.as_str())
                .unwrap_or("markdown");

            format!(
                "Generate {} documentation for:\n\n```\n{}\n```",
                format, code
            )
        }

        _ => anyhow::bail!("Unknown tool: {}", tool_name),
    };

    Ok(prompt)
}

fn handle_resources_list() -> Result<serde_json::Value> {
    debug!("Handling resources/list request");

    Ok(json!({
        "resources": []
    }))
}

fn handle_prompts_list() -> Result<serde_json::Value> {
    debug!("Handling prompts/list request");

    Ok(json!({
        "prompts": [
            {
                "name": "code_assistant",
                "description": "General purpose coding assistant",
                "arguments": []
            }
        ]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_request_parsing() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let request: McpRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.method, "initialize");
        assert_eq!(request.jsonrpc, "2.0");
    }

    #[test]
    fn test_build_code_generation_prompt() {
        let args = json!({
            "description": "parse JSON",
            "language": "python"
        });

        let prompt = build_tool_prompt("code_generation", &args).unwrap();
        assert!(prompt.contains("python"));
        assert!(prompt.contains("parse JSON"));
    }
}
