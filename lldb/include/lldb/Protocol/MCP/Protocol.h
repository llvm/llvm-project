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

struct Error {
  /// The error type that occurred.
  int64_t code = 0;
  /// A short description of the error. The message SHOULD be limited to a
  /// concise single sentence.
  std::string message;
  /// Additional information about the error. The value of this member is
  /// defined by the sender (e.g. detailed error information, nested errors
  /// etc.).
  std::optional<llvm::json::Value> data;
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

struct ToolCapability {
  /// Whether this server supports notifications for changes to the tool list.
  bool listChanged = false;
};

llvm::json::Value toJSON(const ToolCapability &);
bool fromJSON(const llvm::json::Value &, ToolCapability &, llvm::json::Path);

struct ResourceCapability {
  /// Whether this server supports notifications for changes to the resources
  /// list.
  bool listChanged = false;

  ///  Whether subscriptions are supported.
  bool subscribe = false;
};

llvm::json::Value toJSON(const ResourceCapability &);
bool fromJSON(const llvm::json::Value &, ResourceCapability &,
              llvm::json::Path);

/// Capabilities that a server may support. Known capabilities are defined here,
/// in this schema, but this is not a closed set: any server can define its own,
/// additional capabilities.
struct Capabilities {
  /// Tool capabilities of the server.
  ToolCapability tools;

  /// Resource capabilities of the server.
  ResourceCapability resources;
};

llvm::json::Value toJSON(const Capabilities &);
bool fromJSON(const llvm::json::Value &, Capabilities &, llvm::json::Path);

/// A known resource that the server is capable of reading.
struct Resource {
  /// The URI of this resource.
  std::string uri;

  /// A human-readable name for this resource.
  std::string name;

  /// A description of what this resource represents.
  std::string description;

  /// The MIME type of this resource, if known.
  std::string mimeType;
};

llvm::json::Value toJSON(const Resource &);
bool fromJSON(const llvm::json::Value &, Resource &, llvm::json::Path);

/// The contents of a specific resource or sub-resource.
struct ResourceContents {
  /// The URI of this resource.
  std::string uri;

  /// The text of the item. This must only be set if the item can actually be
  /// represented as text (not binary data).
  std::string text;

  /// The MIME type of this resource, if known.
  std::string mimeType;
};

llvm::json::Value toJSON(const ResourceContents &);
bool fromJSON(const llvm::json::Value &, ResourceContents &, llvm::json::Path);

/// The server's response to a resources/read request from the client.
struct ResourceResult {
  std::vector<ResourceContents> contents;
};

llvm::json::Value toJSON(const ResourceResult &);
bool fromJSON(const llvm::json::Value &, ResourceResult &, llvm::json::Path);

/// Text provided to or from an LLM.
struct TextContent {
  /// The text content of the message.
  std::string text;
};

llvm::json::Value toJSON(const TextContent &);
bool fromJSON(const llvm::json::Value &, TextContent &, llvm::json::Path);

struct TextResult {
  std::vector<TextContent> content;
  bool isError = false;
};

llvm::json::Value toJSON(const TextResult &);
bool fromJSON(const llvm::json::Value &, TextResult &, llvm::json::Path);

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

} // namespace lldb_protocol::mcp

#endif
