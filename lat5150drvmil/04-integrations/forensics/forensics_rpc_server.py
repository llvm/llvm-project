#!/usr/bin/env python3
"""
DBXForensics RPC Server for Windows VM

Flask-based HTTP server that receives RPC requests from Dom0 (Linux)
and executes DBXForensics tools on Windows VM.

Install on Windows VM:
    pip install flask

Run:
    python forensics_rpc_server.py

Listens on: http://0.0.0.0:5000
"""

import os
import sys
import json
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:\\Forensics\\rpc_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# DBXForensics tool paths (adjust if needed)
TOOLS_BASE = Path("C:/Forensics/Tools")

TOOLS = {
    'dbxScreenshot': TOOLS_BASE / 'dbxScreenshot.exe',
    'dbxELA': TOOLS_BASE / 'dbxELA.exe',
    'dbxNoiseMap': TOOLS_BASE / 'dbxNoiseMap.exe',
    'dbxMetadata': TOOLS_BASE / 'dbxMetadata.exe',
    'dbxHashFile': TOOLS_BASE / 'dbxHashFile.exe',
    'dbxSeqCheck': TOOLS_BASE / 'dbxSeqCheck.exe',
    'dbxCsvViewer': TOOLS_BASE / 'dbxCsvViewer.exe',
    'dbxGhost': TOOLS_BASE / 'dbxGhost.exe',
    'dbxMouseRecorder': TOOLS_BASE / 'dbxMouseRecorder.exe'
}

# Check which tools are available at startup
AVAILABLE_TOOLS = {}
for tool_name, tool_path in TOOLS.items():
    if tool_path.exists():
        AVAILABLE_TOOLS[tool_name] = str(tool_path)
        logger.info(f"✓ {tool_name} found at {tool_path}")
    else:
        logger.warning(f"✗ {tool_name} not found at {tool_path}")

logger.info(f"Loaded {len(AVAILABLE_TOOLS)}/{len(TOOLS)} tools")


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint

    Returns:
        JSON with server status and available tools
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'tools_loaded': len(AVAILABLE_TOOLS),
        'tools_total': len(TOOLS),
        'tools_available': list(AVAILABLE_TOOLS.keys()),
        'server_version': '1.0.0'
    })


@app.route('/tools', methods=['GET'])
def list_tools():
    """
    List all available forensics tools

    Returns:
        JSON dict of tool names to paths and availability
    """
    tools_info = {}

    for tool_name, tool_path in TOOLS.items():
        exists = tool_path.exists()
        tools_info[tool_name] = {
            'path': str(tool_path),
            'exists': exists,
            'available': tool_name in AVAILABLE_TOOLS
        }

    return jsonify(tools_info)


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Execute forensics tool via RPC

    Request JSON:
        {
            "tool": "dbxELA",
            "input_file": "C:\\Forensics\\input\\screenshot.jpg",
            "args": ["/quality:90"],
            "output_file": "C:\\Forensics\\output\\result.json"  # optional
        }

    Returns:
        JSON with execution results
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        # Extract parameters
        tool_name = data.get('tool')
        input_file = data.get('input_file')
        args = data.get('args', [])
        output_file = data.get('output_file')

        # Validate required parameters
        if not tool_name:
            return jsonify({
                'success': False,
                'error': 'Missing required parameter: tool'
            }), 400

        if not input_file:
            return jsonify({
                'success': False,
                'error': 'Missing required parameter: input_file'
            }), 400

        # Check if tool is available
        if tool_name not in AVAILABLE_TOOLS:
            return jsonify({
                'success': False,
                'error': f'Tool not available: {tool_name}',
                'available_tools': list(AVAILABLE_TOOLS.keys())
            }), 404

        # Get tool path
        tool_path = Path(AVAILABLE_TOOLS[tool_name])

        # Validate input file exists
        input_path = Path(input_file)
        if not input_path.exists():
            return jsonify({
                'success': False,
                'error': f'Input file not found: {input_file}'
            }), 404

        # Build command
        cmd = [str(tool_path), str(input_file)]
        cmd.extend(args)

        logger.info(f"Executing: {' '.join(cmd)}")

        # Execute tool
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(tool_path.parent)
            )

            execution_time = time.time() - start_time

            # Log execution
            logger.info(
                f"Tool: {tool_name} | "
                f"Return code: {result.returncode} | "
                f"Time: {execution_time:.2f}s"
            )

            # Save output to file if requested
            if output_file and result.returncode == 0:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w') as f:
                    json.dump({
                        'tool': tool_name,
                        'input_file': input_file,
                        'returncode': result.returncode,
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'execution_time': execution_time,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)

                logger.info(f"Output saved to: {output_file}")

            # Return results
            return jsonify({
                'success': result.returncode == 0,
                'tool': tool_name,
                'input_file': input_file,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'execution_time': execution_time,
                'output_file': output_file,
                'timestamp': datetime.now().isoformat()
            })

        except subprocess.TimeoutExpired:
            logger.error(f"Tool execution timeout: {tool_name}")
            return jsonify({
                'success': False,
                'error': 'Tool execution timeout (300s)',
                'tool': tool_name
            }), 408

        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return jsonify({
                'success': False,
                'error': f'Tool execution error: {str(e)}',
                'tool': tool_name
            }), 500

    except Exception as e:
        logger.error(f"Request handling error: {e}")
        return jsonify({
            'success': False,
            'error': f'Request handling error: {str(e)}'
        }), 500


@app.route('/batch', methods=['POST'])
def batch_analyze():
    """
    Execute multiple forensics tools in batch

    Request JSON:
        {
            "jobs": [
                {
                    "tool": "dbxELA",
                    "input_file": "C:\\Forensics\\input\\img1.jpg",
                    "args": []
                },
                {
                    "tool": "dbxNoiseMap",
                    "input_file": "C:\\Forensics\\input\\img1.jpg",
                    "args": []
                }
            ]
        }

    Returns:
        JSON with batch execution results
    """
    try:
        data = request.get_json()

        if not data or 'jobs' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required parameter: jobs'
            }), 400

        jobs = data['jobs']
        results = []

        for i, job in enumerate(jobs):
            logger.info(f"Processing batch job {i+1}/{len(jobs)}")

            # Execute each job
            tool_name = job.get('tool')
            input_file = job.get('input_file')
            args = job.get('args', [])

            if not tool_name or not input_file:
                results.append({
                    'success': False,
                    'job_index': i,
                    'error': 'Missing tool or input_file'
                })
                continue

            if tool_name not in AVAILABLE_TOOLS:
                results.append({
                    'success': False,
                    'job_index': i,
                    'error': f'Tool not available: {tool_name}'
                })
                continue

            # Execute tool
            tool_path = Path(AVAILABLE_TOOLS[tool_name])
            cmd = [str(tool_path), str(input_file)]
            cmd.extend(args)

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                results.append({
                    'success': result.returncode == 0,
                    'job_index': i,
                    'tool': tool_name,
                    'input_file': input_file,
                    'returncode': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                })

            except Exception as e:
                results.append({
                    'success': False,
                    'job_index': i,
                    'tool': tool_name,
                    'error': str(e)
                })

        # Calculate summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful

        return jsonify({
            'success': failed == 0,
            'total_jobs': len(results),
            'successful': successful,
            'failed': failed,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Batch execution error: {e}")
        return jsonify({
            'success': False,
            'error': f'Batch execution error: {str(e)}'
        }), 500


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """
    Shutdown RPC server (for maintenance)

    Requires authentication token in production
    """
    logger.info("Shutdown requested")

    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        return jsonify({
            'success': False,
            'error': 'Not running with the Werkzeug Server'
        }), 500

    func()
    return jsonify({
        'success': True,
        'message': 'Server shutting down...'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("DBXForensics RPC Server")
    print("=" * 60)
    print()
    print(f"Tools loaded: {len(AVAILABLE_TOOLS)}/{len(TOOLS)}")
    print()

    for tool_name in AVAILABLE_TOOLS:
        print(f"  ✓ {tool_name}")

    print()
    print("Starting Flask server on http://0.0.0.0:5000")
    print()
    print("Endpoints:")
    print("  GET  /health      - Health check")
    print("  GET  /tools       - List available tools")
    print("  POST /analyze     - Execute forensics tool")
    print("  POST /batch       - Batch execution")
    print("  POST /shutdown    - Shutdown server")
    print()
    print("Press Ctrl+C to stop server")
    print("=" * 60)
    print()

    # Run Flask server
    # host='0.0.0.0' allows access from Dom0
    # port=5000 is the standard Flask port
    app.run(host='0.0.0.0', port=5000, debug=False)
