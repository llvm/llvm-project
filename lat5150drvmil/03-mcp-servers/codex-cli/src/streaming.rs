//! Streaming response handling

use anyhow::{Context, Result};
use futures::Stream;
use pin_project::pin_project;
use reqwest::Response;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use tracing::{debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StreamChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[pin_project]
pub struct ResponseStream {
    #[pin]
    inner: Pin<Box<dyn Stream<Item = Result<String>> + Send>>,
}

impl ResponseStream {
    /// Create new response stream from HTTP response
    pub async fn new(response: Response) -> Result<Self> {
        debug!("Creating response stream");

        let stream = async_stream::stream! {
            let mut bytes_stream = response.bytes_stream();

            let mut buffer = String::new();

            while let Some(chunk_result) = futures::StreamExt::next(&mut bytes_stream).await {
                match chunk_result {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&text);

                        // Process complete lines (SSE format)
                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].trim().to_string();
                            buffer = buffer[newline_pos + 1..].to_string();

                            if line.is_empty() || line.starts_with(':') {
                                continue;
                            }

                            // Parse SSE data line
                            if let Some(data) = line.strip_prefix("data: ") {
                                if data == "[DONE]" {
                                    debug!("Stream completed");
                                    break;
                                }

                                match serde_json::from_str::<StreamChunk>(data) {
                                    Ok(chunk) => {
                                        if let Some(content) = chunk.content {
                                            yield Ok(content);
                                        }

                                        if let Some(error) = chunk.error {
                                            yield Err(anyhow::anyhow!("Stream error: {}", error));
                                            break;
                                        }

                                        if chunk.finish_reason.is_some() {
                                            debug!("Stream finished: {:?}", chunk.finish_reason);
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        warn!("Failed to parse chunk: {} - data: {}", e, data);
                                        // Continue processing other chunks
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(anyhow::anyhow!("Stream read error: {}", e));
                        break;
                    }
                }
            }
        };

        Ok(Self {
            inner: Box::pin(stream),
        })
    }

    /// Get next chunk from stream
    pub async fn next(&mut self) -> Option<Result<String>> {
        use futures::StreamExt;
        self.inner.next().await
    }
}

impl Stream for ResponseStream {
    type Item = Result<String>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();
        this.inner.poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_chunk_parsing() {
        let json = r#"{"content":"Hello","finish_reason":null}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();

        assert_eq!(chunk.content, Some("Hello".to_string()));
        assert!(chunk.finish_reason.is_none());
    }

    #[test]
    fn test_stream_chunk_with_error() {
        let json = r#"{"error":"Rate limit exceeded"}"#;
        let chunk: StreamChunk = serde_json::from_str(json).unwrap();

        assert_eq!(chunk.error, Some("Rate limit exceeded".to_string()));
    }
}
