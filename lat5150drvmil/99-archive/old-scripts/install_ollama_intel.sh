#!/bin/bash
# Ollama Installation - Intel CPU Only (No GPU drivers needed)

set -e

echo "Installing Ollama for Intel CPU (NPU acceleration)..."

# Download Ollama binary directly
cd /tmp
curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
chmod +x ollama

# Install to /usr/local/bin
echo "1786" | sudo -S install -o root -g root -m 755 ollama /usr/local/bin/ollama

# Create ollama user
echo "1786" | sudo -S useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama 2>/dev/null || true

# Create systemd service
cat << 'SERVICE' | sudo tee /etc/systemd/system/ollama.service
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=/usr/local/bin:/usr/bin:/bin"

[Install]
WantedBy=default.target
SERVICE

# Reload systemd and start
echo "1786" | sudo -S systemctl daemon-reload
echo "1786" | sudo -S systemctl enable ollama
echo "1786" | sudo -S systemctl start ollama

sleep 3

# Check status
systemctl status ollama --no-pager | head -15

echo ""
echo "✅ Ollama installed!"
echo ""
echo "Next: Pull a model"
echo "  ollama pull codellama:70b     (Recommended - 39GB)"
echo "  ollama pull llama3.1:70b      (Alternative - 40GB)"
echo "  ollama pull qwen2.5-coder:32b (Smaller - 19GB)"
echo ""
echo "Test:"
echo "  ollama run codellama:70b 'Hello'"
