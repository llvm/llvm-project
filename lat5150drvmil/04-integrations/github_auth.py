#!/usr/bin/env python3
"""
GitHub Authentication Manager
Supports: SSH keys, YubiKey, GPG, no token needed
"""

import subprocess
import os
from pathlib import Path
import json

class GitHubAuth:
    def __init__(self):
        self.ssh_dir = Path.home() / '.ssh'
        self.git_config = Path.home() / '.gitconfig'

    def check_ssh_keys(self):
        """Check for existing SSH keys (including YubiKey)"""
        if not self.ssh_dir.exists():
            return {"status": "no_keys", "keys": []}

        keys = []
        for key_file in ['id_rsa', 'id_ed25519', 'id_ecdsa']:
            pub_key = self.ssh_dir / f'{key_file}.pub'
            if pub_key.exists():
                try:
                    key_content = pub_key.read_text().strip()
                    keys.append({
                        "file": key_file,
                        "type": key_file.split('_')[1],
                        "public_key": key_content[:80] + "...",
                        "yubikey": "cardno:" in key_content.lower()
                    })
                except:
                    pass

        return {"status": "found" if keys else "no_keys", "keys": keys}

    def check_yubikey(self):
        """Check if YubiKey is connected and configured"""
        try:
            # Check for YubiKey via gpg
            result = subprocess.run(
                ['gpg', '--card-status'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and 'Yubico' in result.stdout:
                return {
                    "connected": True,
                    "type": "YubiKey detected",
                    "details": result.stdout
                }
        except:
            pass

        # Check for YubiKey via ykman
        try:
            result = subprocess.run(
                ['ykman', 'info'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return {
                    "connected": True,
                    "type": "YubiKey",
                    "details": result.stdout
                }
        except:
            pass

        return {"connected": False}

    def setup_ssh_agent(self):
        """Setup SSH agent for key-based auth"""
        try:
            # Check if ssh-agent is running
            result = subprocess.run(
                ['ssh-add', '-l'],
                capture_output=True,
                text=True
            )

            loaded_keys = []
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        loaded_keys.append(line)

            return {
                "ssh_agent_running": result.returncode in [0, 1],
                "loaded_keys": loaded_keys,
                "key_count": len(loaded_keys)
            }
        except:
            return {"ssh_agent_running": False}

    def configure_git_ssh(self):
        """Configure git to use SSH instead of HTTPS"""
        commands = [
            "git config --global url.\"git@github.com:\".insteadOf \"https://github.com/\"",
            "git config --global gpg.format ssh",  # Use SSH for signing
        ]

        results = []
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True
                )
                results.append({
                    "command": cmd,
                    "success": result.returncode == 0
                })
            except Exception as e:
                results.append({
                    "command": cmd,
                    "success": False,
                    "error": str(e)
                })

        return {"configured": True, "results": results}

    def test_github_access(self):
        """Test GitHub access via SSH"""
        try:
            result = subprocess.run(
                ['ssh', '-T', 'git@github.com'],
                capture_output=True,
                text=True,
                timeout=10
            )

            # GitHub returns exit code 1 even on success with message
            success = "successfully authenticated" in result.stderr.lower()

            return {
                "accessible": success,
                "message": result.stderr,
                "authenticated": success
            }
        except Exception as e:
            return {
                "accessible": False,
                "error": str(e)
            }

    def get_status(self):
        """Get complete authentication status"""
        return {
            "ssh_keys": self.check_ssh_keys(),
            "yubikey": self.check_yubikey(),
            "ssh_agent": self.setup_ssh_agent(),
            "github_access": self.test_github_access()
        }

    def setup_guide(self):
        """Return setup guide for YubiKey + GitHub"""
        return """
╔══════════════════════════════════════════════════════════════╗
║        GitHub Authentication with YubiKey - Setup Guide      ║
╚══════════════════════════════════════════════════════════════╝

METHOD 1: SSH Key on YubiKey (Recommended)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Generate SSH key on YubiKey:
   ykman piv keys generate -a ECCP256 9a /tmp/public.pem
   ykman piv certificates generate -s "SSH Key" 9a /tmp/public.pem

2. Extract public key:
   ssh-keygen -D /usr/lib/x86_64-linux-gnu/opensc-pkcs11.so -e

3. Add to GitHub:
   - Go to github.com → Settings → SSH Keys
   - Paste the public key
   - Save

4. Configure Git to use YubiKey:
   echo 'PKCS11Provider /usr/lib/x86_64-linux-gnu/opensc-pkcs11.so' >> ~/.ssh/config
   git config --global url."git@github.com:".insteadOf "https://github.com/"

5. Test:
   ssh -T git@github.com

METHOD 2: GPG Key on YubiKey
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Generate GPG key on YubiKey:
   gpg --card-edit
   > admin
   > generate

2. Export public key:
   gpg --armor --export YOUR_KEY_ID

3. Add to GitHub GPG keys

4. Configure Git:
   git config --global user.signingkey YOUR_KEY_ID
   git config --global commit.gpgsign true

METHOD 3: Existing SSH Key (No YubiKey)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Generate key if needed:
   ssh-keygen -t ed25519 -C "your_email@example.com"

2. Add to ssh-agent:
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519

3. Copy public key:
   cat ~/.ssh/id_ed25519.pub

4. Add to GitHub SSH keys

5. Configure Git:
   git config --global url."git@github.com:".insteadOf "https://github.com/"

CURRENT STATUS:
Run this interface to check your setup status.
"""

# CLI
if __name__ == "__main__":
    import sys

    auth = GitHubAuth()

    if len(sys.argv) < 2:
        print("GitHub Auth Manager - Usage:")
        print("  python3 github_auth.py status")
        print("  python3 github_auth.py check-yubikey")
        print("  python3 github_auth.py check-ssh")
        print("  python3 github_auth.py configure")
        print("  python3 github_auth.py test")
        print("  python3 github_auth.py guide")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "status":
        status = auth.get_status()
        print(json.dumps(status, indent=2))

    elif cmd == "check-yubikey":
        yubikey = auth.check_yubikey()
        print(json.dumps(yubikey, indent=2))

    elif cmd == "check-ssh":
        ssh_keys = auth.check_ssh_keys()
        print(json.dumps(ssh_keys, indent=2))

    elif cmd == "configure":
        result = auth.configure_git_ssh()
        print(json.dumps(result, indent=2))

    elif cmd == "test":
        result = auth.test_github_access()
        print(json.dumps(result, indent=2))

    elif cmd == "guide":
        print(auth.setup_guide())
