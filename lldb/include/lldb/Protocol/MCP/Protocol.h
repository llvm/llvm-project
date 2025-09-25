//===- Protocol.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains POD structs based on the MCP specification at
// https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2024-11-05/schema.json
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PROTOCOL_MCP_PROTOCOL_H
#define LLDB_PROTOCOL_MCP_PROTOCOL_H

#include "llvm/Support/JSON.h"
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lldb_protocol::mcp {

static llvm::StringLiteral kProtocolVersion = "2024-11-05";

/// A Request or Response 'id'.
///
/// NOTE: This differs from the JSON-RPC 2.0 spec. The MCP spec says this must
/// be a string or number, excluding a json 'null' as a valid id.
using Id = std::variant<int64_t, std::string>;

/// A request that expects a response.
struct Request {
  /// The request id.
  Id id = 0;
  /// The method to be invoked.
  std::string method;
  /// The method's params.
  std::optional<llvm::json::Value> params;
};
llvm::json::Value toJSON(const Request &);
bool fromJSON(const llvm::json::Value &, Request &, llvm::json::Path);
bool operator==(const Request &, const Request &);

enum ErrorCode : signed {
  /// Invalid JSON was received by the server. An error occurred on the server
  /// while parsing the JSON text.
  eErrorCodeParseError = -32700,
  /// The JSON sent is not a valid Request object.
  eErrorCodeInvalidRequest = -32600,
  /// The method does not exist / is not available.
  eErrorCodeMethodNotFound = -32601,
  /// Invalid method parameter(s).
  eErrorCodeInvalidParams = -32602,
  /// Internal JSON-RPC error.
  eErrorCodeInternalError = -32603,
};

struct Error {
  /// The error type that occurred.
  int64_t code = 0;
  /// A short description of the error. The message SHOULD be limited to a
  /// concise single sentence.
  std::string message;
  /// Additional information about the error. The value of this member is
  /// defined by the sender (e.g. detailed error information, nested errors
  /// etc.).
  std::optional<llvm::json::Value> data = std::nullopt;
};
llvm::json::Value toJSON(const Error &);
bool fromJSON(const llvm::json::Value &, Error &, llvm::json::Path);
bool operator==(const Error &, const Error &);

/// A response to a request, either an error or a result.
struct Response {
  /// The request id.
  Id id = 0;
  /// The result of the request, either an Error or the JSON value of the
  /// response.
  std::variant<Error, llvm::json::Value> result;
};
llvm::json::Value toJSON(const Response &);
bool fromJSON(const llvm::json::Value &, Response &, llvm::json::Path);
bool operator==(const Response &, const Response &);

/// A notification which does not expect a response.
struct Notification {
  /// The method to be invoked.
  std::string method;
  /// The notification's params.
  std::optional<llvm::json::Value> params;
};
llvm::json::Value toJSON(const Notification &);
bool fromJSON(const llvm::json::Value &, Notification &, llvm::json::Path);
bool operator==(const Notification &, const Notification &);

/// A general message as defined by the JSON-RPC 2.0 spec.
using Message = std::variant<Request, Response, Notification>;
// With clang-cl and MSVC STL 202208, convertible can be false later if we do
// not force it to be checked early here.
static_assert(std::is_convertible_v<Message, Message>,
              "Message is not convertible to itself");
bool fromJSON(const llvm::json::Value &, Message &, llvm::json::Path);
llvm::json::Value toJSON(const Message &);

/// A known resource that the server is capable of reading.
struct Resource {
  /// The URI of this resource.
  std::string uri;

  /// A human-readable name for this resource.
  std::string name;

  /// A description of what this resource represents.
  std::string description = "";

  /// The MIME type of this resource, if known.
  std::string mimeType = "";
};

llvm::json::Value toJSON(const Resource &);
bool fromJSON(const llvm::json::Value &, Resource &, llvm::json::Path);

/// The server’s response to a resources/list request from the client.
struct ListResourcesResult {
  std::vector<Resource> resources;
};
llvm::json::Value toJSON(const ListResourcesResult &);
bool fromJSON(const llvm::json::Value &, ListResourcesResult &,
              llvm::json::Path);

/// The contents of a specific resource or sub-resource.
struct TextResourceContents {
  /// The URI of this resource.
  std::string uri;

  /// The text of the item. This must only be set if the item can actually be
  /// represented as text (not binary data).
  std::string text;

  /// The MIME type of this resource, if known.
  std::string mimeType;
};

llvm::json::Value toJSON(const TextResourceContents &);
bool fromJSON(const llvm::json::Value &, TextResourceContents &,
              llvm::json::Path);

/// Sent from the client to the server, to read a specific resource URI.
struct ReadResourceParams {
  /// The URI of the resource to read. The URI can use any protocol; it is up to
  /// the server how to interpret it.
  std::string uri;
};
llvm::json::Value toJSON(const ReadResourceParams &);
bool fromJSON(const llvm::json::Value &, ReadResourceParams &,
              llvm::json::Path);

/// The server's response to a resources/read request from the client.
struct ReadResourceResult {
  std::vector<TextResourceContents> contents;
};
llvm::json::Value toJSON(const ReadResourceResult &);
bool fromJSON(const llvm::json::Value &, ReadResourceResult &,
              llvm::json::Path);

/// Text provided to or from an LLM.
struct TextContent {
  /// The text content of the message.
  std::string text;
};
llvm::json::Value toJSON(const TextContent &);
bool fromJSON(const llvm::json::Value &, TextContent &, llvm::json::Path);

/// Definition for a tool the client can call.
struct ToolDefinition {
  /// Unique identifier for the tool.
  std::string name;

  /// Human-readable description.
  std::string description;

  // JSON Schema for the tool's parameters.
  std::optional<llvm::json::Value> inputSchema;
};
llvm::json::Value toJSON(const ToolDefinition &);
bool fromJSON(const llvm::json::Value &, ToolDefinition &, llvm::json::Path);

using ToolArguments = std::variant<std::monostate, llvm::json::Value>;

/// Describes the name and version of an MCP implementation, with an optional
/// title for UI representation.
struct Implementation {
  /// Intended for programmatic or logical use, but used as a display name in
  /// past specs or fallback (if title isn’t present).
  std::string name;

  std::string version;

  /// Intended for UI and end-user contexts — optimized to be human-readable and
  /// easily understood, even by those unfamiliar with domain-specific
  /// terminology.
  ///
  /// If not provided, the name should be used for display (except for Tool,
  /// where annotations.title should be given precedence over using name, if
  /// present).
  std::string title = "";
};
llvm::json::Value toJSON(const Implementation &);
bool fromJSON(const llvm::json::Value &, Implementation &, llvm::json::Path);

/// Capabilities a client may support. Known capabilities are defined here, in
/// this schema, but this is not a closed set: any client can define its own,
/// additional capabilities.
struct ClientCapabilities {};
llvm::json::Value toJSON(const ClientCapabilities &);
bool fromJSON(const llvm::json::Value &, ClientCapabilities &,
              llvm::json::Path);

/// Capabilities that a server may support. Known capabilities are defined here,
/// in this schema, but this is not a closed set: any server can define its own,
/// additional capabilities.
struct ServerCapabilities {
  bool supportsToolsList = false;
  bool supportsResourcesList = false;
  bool supportsResourcesSubscribe = false;

  /// Utilities.
  bool supportsCompletions = false;
  bool supportsLogging = false;
};
llvm::json::Value toJSON(const ServerCapabilities &);
bool fromJSON(const llvm::json::Value &, ServerCapabilities &,
              llvm::json::Path);

/// Initialization

/// This request is sent from the client to the server when it first connects,
/// asking it to begin initialization.
struct InitializeParams {
  /// The latest version of the Model Context Protocol that the client supports.
  /// The client MAY decide to support older versions as well.
  std::string protocolVersion;

  ClientCapabilities capabilities;

  Implementation clientInfo;
};
llvm::json::Value toJSON(const InitializeParams &);
bool fromJSON(const llvm::json::Value &, InitializeParams &, llvm::json::Path);

/// After receiving an initialize request from the client, the server sends this
/// response.
struct InitializeResult {
  /// The version of the Model Context Protocol that the server wants to use.
  /// This may not match the version that the client requested. If the client
  /// cannot support this version, it MUST disconnect.
  std::string protocolVersion;

  ServerCapabilities capabilities;
  Implementation serverInfo;

  /// Instructions describing how to use the server and its features.
  ///
  /// This can be used by clients to improve the LLM's understanding of
  /// available tools, resources, etc. It can be thought of like a "hint" to the
  /// model. For example, this information MAY be added to the system prompt.
  std::string instructions = "";
};
llvm::json::Value toJSON(const InitializeResult &);
bool fromJSON(const llvm::json::Value &, InitializeResult &, llvm::json::Path);

/// Special case parameter or result that has no value.
using Void = std::monostate;
llvm::json::Value toJSON(const Void &);
bool fromJSON(const llvm::json::Value &, Void &, llvm::json::Path);

/// The server's response to a `tools/list` request from the client.
struct ListToolsResult {
  std::vector<ToolDefinition> tools;
};
llvm::json::Value toJSON(const ListToolsResult &);
bool fromJSON(const llvm::json::Value &, ListToolsResult &, llvm::json::Path);

/// Supported content types, currently only TextContent, but the spec includes
/// additional content types.
using ContentBlock = TextContent;

/// Used by the client to invoke a tool provided by the server.
struct CallToolParams {
  std::string name;
  std::optional<llvm::json::Value> arguments;
};
llvm::json::Value toJSON(const CallToolParams &);
bool fromJSON(const llvm::json::Value &, CallToolParams &, llvm::json::Path);

/// The server’s response to a tool call.
struct CallToolResult {
  /// A list of content objects that represent the unstructured result of the
  /// tool call.
  std::vector<ContentBlock> content;

  /// Whether the tool call ended in an error.
  ///
  /// If not set, this is assumed to be false (the call was successful).
  ///
  /// Any errors that originate from the tool SHOULD be reported inside the
  /// result object, with `isError` set to true, not as an MCP protocol-level
  /// error response. Otherwise, the LLM would not be able to see that an error
  /// occurred and self-correct.
  ///
  /// However, any errors in finding the tool, an error indicating that the
  /// server does not support tool calls, or any other exceptional conditions,
  /// should be reported as an MCP error response.
  bool isError = false;

  /// An optional JSON object that represents the structured result of the tool
  /// call.
  std::optional<llvm::json::Value> structuredContent = std::nullopt;
};
llvm::json::Value toJSON(const CallToolResult &);
bool fromJSON(const llvm::json::Value &, CallToolResult &, llvm::json::Path);

} // namespace lldb_protocol::mcp

#endif
