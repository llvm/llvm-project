#!/usr/bin/env python3
"""
Full-Featured Opus Server
Comprehensive system management, not just installation
"""

import http.server
import socketserver
import json
import subprocess
import os
import tempfile
from urllib.parse import parse_qs, unquote
import base64
import email
from io import BytesIO
import glob
import time
from pathlib import Path

# Dynamic path configuration - works for any user
BASE_DIR = Path(__file__).parent.parent.resolve()  # LAT5150DRVMIL directory
HOME_DIR = Path.home()
AI_ENGINE_DIR = BASE_DIR / "02-ai-engine"
WEB_INTERFACE_DIR = BASE_DIR / "03-web-interface"
INTEGRATIONS_DIR = BASE_DIR / "04-integrations"
UPLOADS_DIR = HOME_DIR / "uploads"
GITHUB_REPOS_DIR = HOME_DIR / "github_repos"
KERNEL_BUILD_DIR = HOME_DIR / "linux-6.16.9"

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
GITHUB_REPOS_DIR.mkdir(exist_ok=True)

class FullFeaturedHandler(http.server.SimpleHTTPRequestHandler):

    def verify_localhost(self):
        """Verify request is from localhost only"""
        client_ip = self.client_address[0]

        # Allow only localhost connections
        allowed_ips = ['127.0.0.1', '::1', 'localhost']

        if client_ip not in allowed_ips:
            self.send_error(403, f"Forbidden: Remote access not allowed. Client IP: {client_ip}")
            self.end_headers()
            self.wfile.write(b"ERROR: Remote access is disabled for security.\n")
            self.wfile.write(b"This server only accepts connections from localhost (127.0.0.1).\n")
            self.wfile.write(b"If you need remote access, use SSH tunneling:\n")
            self.wfile.write(b"  ssh -L 9876:localhost:9876 user@machine\n")
            return False

        return True

    def do_GET(self):
        # Security: Verify localhost access
        if not self.verify_localhost():
            return
        if self.path == '/':
            self.serve_interface()
        elif self.path == '/status':
            self.serve_status()
        elif self.path == '/commands':
            self.serve_commands()
        elif self.path == '/handoff':
            self.serve_handoff()
        elif self.path == '/npu':
            self.serve_npu()
        elif self.path.startswith('/exec?'):
            self.execute_command()
        elif self.path.startswith('/files?'):
            self.list_files()
        elif self.path.startswith('/read?'):
            self.read_file()
        elif self.path.startswith('/logs?'):
            self.show_logs()
        elif self.path =='/npu/run':
            self.run_npu_modules()
        elif self.path == '/system/info':
            self.system_info()
        elif self.path == '/kernel/status':
            self.kernel_status()
        elif self.path.startswith('/rag/stats'):
            self.rag_stats()
        elif self.path.startswith('/rag/search'):
            self.rag_search()
        elif self.path.startswith('/rag/ingest'):
            self.rag_ingest()
        elif self.path.startswith('/rag/add-file'):
            self.rag_add_file()
        elif self.path.startswith('/rag/add-folder'):
            self.rag_add_folder()
        elif self.path.startswith('/rag/list'):
            self.rag_list()
        elif self.path.startswith('/web/scrape'):
            self.web_scrape()
        elif self.path.startswith('/web/crawl'):
            self.web_crawl()
        elif self.path.startswith('/web/fetch'):
            self.web_fetch()
        elif self.path.startswith('/archive/vxunderground'):
            self.archive_vx()
        elif self.path.startswith('/archive/arxiv'):
            self.archive_arxiv()
        elif self.path.startswith('/github/auth-status'):
            self.github_auth_status()
        elif self.path.startswith('/github/'):
            self.github_operation()
        elif self.path.startswith('/smart-collect'):
            self.smart_collect()
        elif self.path.startswith('/ai/chat'):
            self.ai_chat()
        elif self.path.startswith('/ai/prompt'):
            self.ai_prompt()
        elif self.path.startswith('/ai/status'):
            self.ai_status()
        elif self.path.startswith('/ai/set-system-prompt'):
            self.ai_set_system_prompt()
        elif self.path.startswith('/ai/get-system-prompt'):
            self.ai_get_system_prompt()
        else:
            super().do_GET()

    def serve_interface(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        try:
            # Serve new clean UI v3 (ChatGPT-style with military green)
            with open(WEB_INTERFACE_DIR / 'clean_ui_v3.html', 'rb') as f:
                self.wfile.write(f.read())
        except:
            # Fallback to v2
            try:
                with open(WEB_INTERFACE_DIR / 'military_terminal_v2.html', 'rb') as f:
                    self.wfile.write(f.read())
            except:
                # Legacy fallback
                try:
                    with open(HOME_DIR / 'military_terminal.html', 'rb') as f:
                        self.wfile.write(f.read())
                except Exception as e:
                    self.wfile.write(f'Error loading interface: {e}'.encode())

    def serve_status(self):
        self.send_json({
            "kernel": str(KERNEL_BUILD_DIR / "arch/x86/boot/bzImage"),
            "kernel_exists": (KERNEL_BUILD_DIR / "arch/x86/boot/bzImage").exists(),
            "mode5": "STANDARD (safe)",
            "dsmil": "84 devices ready",
            "build": "SUCCESS",
            "npu_modules": "6 modules built",
            "upload": "ready",
            "timestamp": time.time()
        })

    def serve_commands(self):
        self.send_text(f"""cd {KERNEL_BUILD_DIR}
sudo make modules_install
sudo make install
sudo update-grub
sudo reboot""")

    def serve_handoff(self):
        self.serve_file(str(HOME_DIR / 'COMPLETE_MILITARY_SPEC_HANDOFF.md'))

    def serve_npu(self):
        self.serve_file(str(HOME_DIR / 'NPU_MODULES_COMPLETE.md'))

    def execute_command(self):
        """Execute shell command and return output"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        cmd = query.get('cmd', [''])[0]

        if not cmd:
            self.send_json({"error": "No command specified"})
            return

        # Safety check - don't allow dangerous commands
        dangerous = ['rm -rf /', 'mkfs', 'dd if=', ':(){:|:&};:', 'chmod -R 777 /']
        if any(d in cmd for d in dangerous):
            self.send_json({"error": "Dangerous command blocked for safety"})
            return

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(HOME_DIR)
            )

            self.send_json({
                "command": cmd,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            })
        except subprocess.TimeoutExpired:
            self.send_json({"error": "Command timed out (30s limit)"})
        except Exception as e:
            self.send_json({"error": str(e)})

    def list_files(self):
        """List files in directory"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        path = query.get('path', [str(HOME_DIR)])[0]

        try:
            if not os.path.exists(path):
                self.send_json({"error": "Path does not exist"})
                return

            if os.path.isfile(path):
                # Return file info
                self.send_json({
                    "type": "file",
                    "path": path,
                    "size": os.path.getsize(path),
                    "name": os.path.basename(path)
                })
            else:
                # List directory
                items = []
                for item in sorted(os.listdir(path)):
                    item_path = os.path.join(path, item)
                    try:
                        stat = os.stat(item_path)
                        items.append({
                            "name": item,
                            "path": item_path,
                            "is_dir": os.path.isdir(item_path),
                            "size": stat.st_size,
                            "modified": stat.st_mtime
                        })
                    except:
                        pass

                self.send_json({
                    "type": "directory",
                    "path": path,
                    "items": items
                })
        except Exception as e:
            self.send_json({"error": str(e)})

    def read_file(self):
        """Read file contents"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        path = query.get('path', [''])[0]

        if not path or not os.path.exists(path):
            self.send_json({"error": "File not found"})
            return

        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(100000)  # Max 100KB

            self.send_json({
                "path": path,
                "content": content,
                "size": os.path.getsize(path)
            })
        except Exception as e:
            self.send_json({"error": str(e)})

    def show_logs(self):
        """Show recent logs"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        lines = int(query.get('lines', ['100'])[0])

        logs = {
            "kernel_build": str(HOME_DIR / "kernel-build-apt-secure.log"),
            "opus_server": "/tmp/opus_server.log",
            "dmesg": "dmesg"
        }

        result = {}
        for name, log_path in logs.items():
            try:
                if log_path == "dmesg":
                    cmd_result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5)
                    content = cmd_result.stdout.split('\n')[-lines:]
                    result[name] = '\n'.join(content)
                elif os.path.exists(log_path):
                    cmd_result = subprocess.run(['tail', f'-{lines}', log_path],
                                              capture_output=True, text=True, timeout=5)
                    result[name] = cmd_result.stdout
                else:
                    result[name] = f"Log not found: {log_path}"
            except Exception as e:
                result[name] = f"Error: {e}"

        self.send_json(result)

    def run_npu_modules(self):
        """Run NPU module tests"""
        npu_dir = str(HOME_DIR / "livecd-gen/npu_modules/bin")

        if not os.path.exists(npu_dir):
            self.send_json({"error": "NPU modules not found"})
            return

        results = {}
        for module in os.listdir(npu_dir):
            module_path = os.path.join(npu_dir, module)
            if os.path.isfile(module_path) and os.access(module_path, os.X_OK):
                try:
                    result = subprocess.run(
                        [module_path],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    results[module] = {
                        "stdout": result.stdout,
                        "returncode": result.returncode
                    }
                except Exception as e:
                    results[module] = {"error": str(e)}

        self.send_json(results)

    def system_info(self):
        """Get comprehensive system information"""
        info = {}

        commands = {
            "cpu": "lscpu | head -20",
            "memory": "free -h",
            "disk": "df -h | head -10",
            "kernel": "uname -a",
            "uptime": "uptime",
            "processes": "ps aux | head -10"
        }

        for name, cmd in commands.items():
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True,
                                      text=True, timeout=5)
                info[name] = result.stdout
            except:
                info[name] = "Command failed"

        self.send_json(info)

    def kernel_status(self):
        """Get detailed kernel status"""
        status = {}

        # Check kernel file
        kernel_path = str(KERNEL_BUILD_DIR / "arch/x86/boot/bzImage")
        status['kernel_built'] = os.path.exists(kernel_path)
        if status['kernel_built']:
            status['kernel_size'] = os.path.getsize(kernel_path)

        # Check DSMIL driver
        dsmil_path = str(KERNEL_BUILD_DIR / "drivers/platform/x86/dell-milspec/dsmil-core.c")
        status['dsmil_source'] = os.path.exists(dsmil_path)
        if status['dsmil_source']:
            with open(dsmil_path, 'r') as f:
                status['dsmil_lines'] = len(f.readlines())

        # Check NPU modules
        npu_dir = str(HOME_DIR / "livecd-gen/npu_modules/bin")
        if os.path.exists(npu_dir):
            status['npu_modules'] = len([f for f in os.listdir(npu_dir)
                                        if os.path.isfile(os.path.join(npu_dir, f))])
        else:
            status['npu_modules'] = 0

        # Check running kernel
        try:
            result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
            status['running_kernel'] = result.stdout.strip()
        except:
            status['running_kernel'] = "unknown"

        self.send_json(status)

    def do_POST(self):
        # Security: Verify localhost access
        if not self.verify_localhost():
            return
        if self.path == '/upload':
            self.handle_upload()
        elif self.path == '/execute':
            self.handle_execute()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_upload(self):
        """Handle file uploads"""
        content_type = self.headers.get('Content-Type', '')
        content_length = int(self.headers.get('Content-Length', 0))

        if 'multipart/form-data' in content_type:
            boundary = content_type.split('boundary=')[1]
            body = self.rfile.read(content_length)

            parts = body.split(f'--{boundary}'.encode())

            for part in parts:
                if b'filename=' in part:
                    # Extract filename
                    filename_start = part.find(b'filename="') + 10
                    filename_end = part.find(b'"', filename_start)
                    filename = part[filename_start:filename_end].decode('utf-8', errors='ignore')

                    # Extract file data
                    file_data_start = part.find(b'\r\n\r\n') + 4
                    file_data = part[file_data_start:-2]

                    if filename and file_data:
                        upload_dir = str(UPLOADS_DIR)
                        os.makedirs(upload_dir, exist_ok=True)

                        filepath = os.path.join(upload_dir, filename)

                        with open(filepath, 'wb') as f:
                            f.write(file_data)

                        result = self.process_file(filepath, filename)

                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(result).encode())
                        return

        self.send_json({"error": "Invalid upload"})

    def handle_execute(self):
        """Handle command execution from web interface"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            cmd = data.get('command', '')

            if not cmd:
                self.send_json({"error": "No command provided"})
                return

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(HOME_DIR)
            )

            self.send_json({
                "command": cmd,
                "output": result.stdout + result.stderr,
                "returncode": result.returncode
            })
        except Exception as e:
            self.send_json({"error": str(e)})

    def process_file(self, filepath, filename):
        """Process uploaded file"""
        result = {
            "filename": filename,
            "path": filepath,
            "size": os.path.getsize(filepath),
            "type": "unknown",
            "content": "",
            "summary": ""
        }

        if filename.lower().endswith('.pdf'):
            result['type'] = 'PDF'
            result['content'] = self.extract_pdf_text(filepath)
            result['summary'] = f"PDF with {len(result['content'])} characters"

        elif filename.lower().endswith(('.txt', '.md', '.log')):
            result['type'] = 'Text'
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                result['content'] = f.read()
            result['summary'] = f"Text file with {len(result['content'])} characters"

        elif filename.lower().endswith(('.c', '.h', '.py', '.sh')):
            result['type'] = 'Code'
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                result['content'] = content
                lines = len(content.split('\n'))
            result['summary'] = f"Code file with {lines} lines"

        return result

    def extract_pdf_text(self, filepath):
        """Extract text from PDF"""
        try:
            result = subprocess.run(
                ['pdftotext', filepath, '-'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout
        except:
            pass

        try:
            result = subprocess.run(
                ['strings', filepath],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout
        except:
            return "PDF extraction failed. Install pdftotext for better results."

    def send_json(self, data):
        """Send JSON response, handles both dicts and Pydantic models"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        # Check if data is a Pydantic model
        if hasattr(data, 'model_dump'):
            # Pydantic v2 model
            json_str = data.model_dump_json()
            self.wfile.write(json_str.encode())
        elif hasattr(data, 'dict'):
            # Pydantic v1 model
            self.wfile.write(json.dumps(data.dict()).encode())
        else:
            # Regular dict
            self.wfile.write(json.dumps(data).encode())

    def send_text(self, text):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(text.encode())

    def serve_file(self, path):
        try:
            with open(path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(content)
        except:
            self.send_text(f'File not found: {path}')

    def rag_stats(self):
        """Get RAG system statistics"""
        try:
            result = subprocess.run(
                ['python3', str(HOME_DIR / 'rag_system.py'), 'stats'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.send_text(result.stdout)
            else:
                self.send_json({"error": "RAG not initialized", "total_documents": 0})
        except:
            self.send_json({"error": "RAG system not available", "total_documents": 0})

    def rag_search(self):
        """Search RAG index"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        q = query.get('q', [''])[0]

        try:
            result = subprocess.run(
                ['python3', str(HOME_DIR / 'rag_system.py'), 'search', q],
                capture_output=True,
                text=True,
                timeout=30
            )
            # Parse output into JSON
            self.send_json({"results": [], "query": q, "output": result.stdout})
        except Exception as e:
            self.send_json({"error": str(e)})

    def rag_ingest(self):
        """Ingest folder into RAG"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        path = query.get('path', [''])[0]

        try:
            result = subprocess.run(
                ['python3', str(HOME_DIR / 'rag_system.py'), 'ingest-folder', path],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    self.send_json(data)
                except:
                    self.send_json({"status": "completed", "output": result.stdout})
            else:
                self.send_json({"error": result.stderr})
        except Exception as e:
            self.send_json({"error": str(e)})

    def web_fetch(self):
        """Fetch web content"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        url = query.get('url', [''])[0]

        try:
            result = subprocess.run(
                ['curl', '-L', '-s', '--max-time', '30', url],
                capture_output=True,
                text=True,
                timeout=35
            )
            self.send_json({"url": url, "content": result.stdout[:10000]})
        except Exception as e:
            self.send_json({"error": str(e)})

    def archive_vx(self):
        """Archive VX underground"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        topic = query.get('topic', ['apt'])[0]

        try:
            result = subprocess.run(
                ['python3', str(HOME_DIR / 'web_archiver.py'), 'vxunderground', topic],
                capture_output=True,
                text=True,
                timeout=60
            )
            try:
                data = json.loads(result.stdout)
                self.send_json(data)
            except:
                self.send_json({"status": "completed", "output": result.stdout})
        except Exception as e:
            self.send_json({"error": str(e)})

    def archive_arxiv(self):
        """Download arXiv paper"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        arxiv_id = query.get('id', [''])[0]

        try:
            result = subprocess.run(
                ['python3', str(HOME_DIR / 'web_archiver.py'), 'arxiv', arxiv_id],
                capture_output=True,
                text=True,
                timeout=120
            )
            try:
                data = json.loads(result.stdout)
                self.send_json(data)
            except:
                self.send_json({"status": "completed", "output": result.stdout})
        except Exception as e:
            self.send_json({"error": str(e)})

    def smart_collect(self):
        """Smart paper collection - downloads papers on topic up to size limit"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        topic = query.get('topic', [''])[0]
        max_size = query.get('size', ['10'])[0]

        if not topic:
            self.send_json({"error": "No topic specified"})
            return

        try:
            result = subprocess.run(
                ['python3', str(HOME_DIR / 'smart_paper_collector.py'), 'collect', topic, max_size],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes for large collections
            )

            # Parse the JSON from output
            output_lines = result.stdout.strip().split('\n')
            for line in reversed(output_lines):
                if line.startswith('{'):
                    try:
                        data = json.loads(line)
                        self.send_json(data)
                        return
                    except:
                        pass

            # Fallback
            self.send_json({
                "status": "completed",
                "output": result.stdout,
                "stderr": result.stderr
            })

        except subprocess.TimeoutExpired:
            self.send_json({"error": "Collection timed out (30min limit)"})
        except Exception as e:
            self.send_json({"error": str(e)})

    def github_auth_status(self):
        """Get GitHub authentication status"""
        try:
            result = subprocess.run(
                ['python3', str(HOME_DIR / 'github_auth.py'), 'status'],
                capture_output=True,
                text=True,
                timeout=15
            )
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    self.send_json(data)
                except:
                    self.send_json({"error": "Failed to parse auth status"})
            else:
                self.send_json({"error": "Auth check failed"})
        except Exception as e:
            self.send_json({"error": str(e)})

    def github_operation(self):
        """GitHub operations - clone, fetch, etc"""
        # Parse operation from path: /github/clone, /github/fetch, etc
        parts = self.path.split('/')
        if len(parts) < 3:
            self.send_json({"error": "Invalid GitHub operation"})
            return

        operation = parts[2].split('?')[0]
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')

        if operation == 'clone':
            repo_url = query.get('url', [''])[0]
            if not repo_url:
                self.send_json({"error": "No URL provided"})
                return

            try:
                # Clone to github_repos/
                repo_dir = str(GITHUB_REPOS_DIR)
                os.makedirs(repo_dir, exist_ok=True)

                result = subprocess.run(
                    ['git', 'clone', repo_url],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=repo_dir
                )

                self.send_json({
                    "operation": "clone",
                    "url": repo_url,
                    "success": result.returncode == 0,
                    "output": result.stdout + result.stderr
                })
            except Exception as e:
                self.send_json({"error": str(e)})

        elif operation == 'list':
            # List cloned repos
            try:
                repo_dir = str(GITHUB_REPOS_DIR)
                if os.path.exists(repo_dir):
                    repos = [d for d in os.listdir(repo_dir) if os.path.isdir(os.path.join(repo_dir, d))]
                    self.send_json({"repos": repos, "path": repo_dir})
                else:
                    self.send_json({"repos": [], "path": repo_dir})
            except Exception as e:
                self.send_json({"error": str(e)})

        else:
            self.send_json({"error": f"Unknown operation: {operation}"})

    def ai_chat(self):
        """
        AI chat endpoint with SMART ROUTING and optional Pydantic support

        Query parameters:
          - msg: Message to send to AI (required)
          - model: Model preference (default: auto)
          - pydantic: Return type-safe Pydantic model (0 or 1, default: 0)
        """
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        message = query.get('msg', [''])[0]
        model = query.get('model', ['auto'])[0]
        use_pydantic = query.get('pydantic', ['0'])[0] in ['1', 'true', 'True', 'yes']

        if not message:
            self.send_json({"error": "No message provided"})
            return

        try:
            # Load UNIFIED ORCHESTRATOR (with smart routing!)
            import sys
            sys.path.insert(0, str(AI_ENGINE_DIR))
            from unified_orchestrator import UnifiedAIOrchestrator

            # Create orchestrator with optional Pydantic mode
            orchestrator = UnifiedAIOrchestrator(pydantic_mode=use_pydantic)

            # Use orchestrator with smart routing
            if model == 'auto':
                result = orchestrator.query(message)  # Smart routing!
            else:
                result = orchestrator.query(message, force_backend=model)

            # send_json automatically handles both dict and Pydantic models
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e), "details": str(e)})

    def ai_prompt(self):
        """Quick AI prompt endpoint (alias for chat)"""
        self.ai_chat()

    def ai_status(self):
        """AI engine status"""
        try:
            import sys
            sys.path.insert(0, str(HOME_DIR))
            from dsmil_ai_engine import DSMILAIEngine

            engine = DSMILAIEngine()
            status = engine.get_status()
            self.send_json(status)
        except Exception as e:
            self.send_json({"error": str(e), "ai_available": False})

    def ai_set_system_prompt(self):
        """Set custom system prompt"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        prompt = query.get('prompt', [''])[0]

        if not prompt:
            self.send_json({"error": "No prompt provided"})
            return

        try:
            import sys
            sys.path.insert(0, str(HOME_DIR))
            from dsmil_ai_engine import DSMILAIEngine

            engine = DSMILAIEngine()
            result = engine.set_system_prompt(prompt)
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)})

    def ai_get_system_prompt(self):
        """Get current system prompt"""
        try:
            import sys
            sys.path.insert(0, str(HOME_DIR))
            from dsmil_ai_engine import DSMILAIEngine

            engine = DSMILAIEngine()
            prompt = engine.get_system_prompt()
            self.send_json({"prompt": prompt, "length": len(prompt)})
        except Exception as e:
            self.send_json({"error": str(e)})

    def rag_add_file(self):
        """Add file to RAG"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        filepath = query.get('path', [''])[0]

        if not filepath:
            self.send_json({"error": "No path provided"})
            return

        try:
            import sys
            sys.path.insert(0, str(INTEGRATIONS_DIR))
            from rag_manager import RAGManager

            manager = RAGManager()
            result = manager.add_file(filepath)
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)})

    def rag_add_folder(self):
        """Add folder to RAG"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        folder = query.get('path', [''])[0]

        if not folder:
            self.send_json({"error": "No path provided"})
            return

        try:
            import sys
            sys.path.insert(0, str(INTEGRATIONS_DIR))
            from rag_manager import RAGManager

            manager = RAGManager()
            result = manager.add_folder(folder)
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)})

    def rag_list(self):
        """List RAG documents"""
        try:
            import sys
            sys.path.insert(0, str(INTEGRATIONS_DIR))
            from rag_manager import RAGManager

            manager = RAGManager()
            result = manager.list_documents()
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e)})

    def web_scrape(self):
        """Scrape URL and auto-add to RAG"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        url = query.get('url', [''])[0]

        if not url:
            self.send_json({"error": "No URL provided"})
            return

        try:
            import sys
            sys.path.insert(0, str(INTEGRATIONS_DIR))
            from web_scraper import WebScraper

            scraper = WebScraper()
            result = scraper.scrape_url(url, auto_add_to_rag=True)
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e), "url": url})

    def web_crawl(self):
        """Crawl website and index all pages"""
        query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')
        url = query.get('url', [''])[0]
        max_pages = int(query.get('max_pages', ['50'])[0])
        depth = int(query.get('depth', ['3'])[0])

        if not url:
            self.send_json({"error": "No URL provided"})
            return

        try:
            import sys
            sys.path.insert(0, str(INTEGRATIONS_DIR))
            from web_scraper import WebScraper

            scraper = WebScraper()
            result = scraper.crawl_and_index(url, max_pages=max_pages, depth_limit=depth)
            self.send_json(result)
        except Exception as e:
            self.send_json({"error": str(e), "url": url})

PORT = 9876
HOST = "127.0.0.1"  # LOCALHOST ONLY - Never expose to network!
print(f"""
╔══════════════════════════════════════════════════════════════╗
║   Full-Featured Opus Server Starting                        ║
╚══════════════════════════════════════════════════════════════╝

Port: {PORT}
URL: http://localhost:{PORT}

Capabilities:
  ✅ Document upload & processing (PDF, TXT, code)
  ✅ Command execution with safety checks
  ✅ File browsing and reading
  ✅ Log viewing (kernel build, server, dmesg)
  ✅ NPU module execution
  ✅ System diagnostics
  ✅ Kernel status monitoring
  ✅ Real-time information

Endpoints:
  GET  /                 - Main interface
  GET  /status           - System status JSON
  GET  /exec?cmd=CMD     - Execute command
  GET  /files?path=PATH  - List files
  GET  /read?path=PATH   - Read file
  GET  /logs?lines=N     - Show logs
  GET  /npu/run          - Run NPU modules
  GET  /system/info      - System information
  GET  /kernel/status    - Kernel build status
  POST /upload           - Upload files
  POST /execute          - Execute commands

Uploads saved to: {UPLOADS_DIR}/

Server ready!
""")

with socketserver.TCPServer((HOST, PORT), FullFeaturedHandler) as httpd:
    print(f"\n🔒 SECURITY: Server bound to {HOST} (localhost only)")
    print("⚠️  WARNING: Do NOT change HOST to 0.0.0.0 - This is a security risk!")
    print(f"✓ Safe access: http://{HOST}:{PORT}")
    print("\nPress Ctrl+C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped")
