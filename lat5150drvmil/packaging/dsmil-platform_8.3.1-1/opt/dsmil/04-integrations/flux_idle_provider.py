#!/usr/bin/env python3
"""
Flux Network Idle Provider
Allocate spare cycles to Flux when system idle, reclaim for research instantly
"""
import subprocess
import time
import psutil

class FluxIdleProvider:
    def __init__(self):
        self.idle_threshold_cpu = 20  # % CPU usage
        self.idle_threshold_ram = 50  # % RAM usage
        self.check_interval = 60  # seconds
        self.flux_process = None

    def get_user_idle_time(self):
        """Get minutes since last user input (keyboard/mouse)"""
        try:
            # Check X11 idle time
            result = subprocess.run(['xprintidle'], capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                ms = int(result.stdout.strip())
                return ms / 1000 / 60  # Convert to minutes
        except:
            pass
        return 0

    def get_allocation_level(self):
        """
        Determine resource allocation based on user presence
        Returns: 'minimal', 'moderate', 'aggressive'
        """
        idle_min = self.get_user_idle_time()
        cpu = psutil.cpu_percent(interval=2)

        # User actively present
        if idle_min < 5:
            return 'minimal'  # LP E-cores only

        # User idle but system active
        elif idle_min < 15:
            if cpu < 20:
                return 'moderate'  # LP + E-cores
            else:
                return 'minimal'  # Research active

        # User away (15+ min no input)
        else:
            # Check for AI processes
            for proc in psutil.process_iter(['name', 'cmdline']):
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'ollama' in cmdline.lower():
                    return 'moderate'  # AI running, don't take P-cores

            return 'aggressive'  # User away, no AI, use everything

    def start_flux(self):
        """Start Flux provider with limited resources"""
        if self.flux_process and self.flux_process.poll() is None:
            return  # Already running

        # Allocate: E-cores only, 16GB RAM max
        # Reserve: P-cores, NPU, GPU, NCS2 for research
        cmd = [
            'taskset', '-c', '12-19',  # E-cores only
            'flux-provider',
            '--memory', '16GB',
            '--cpu-threads', '8',
            '--wallet', 'YOUR_FLUX_WALLET'
        ]

        self.flux_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        print(f"FLUX: Started (PID {self.flux_process.pid})")
        print(f"ALLOCATED: E-cores 12-19, 16GB RAM")
        print(f"RESERVED: P-cores, NPU, GPU, NCS2 for research")

    def stop_flux(self):
        """Stop Flux provider immediately"""
        if self.flux_process and self.flux_process.poll() is None:
            self.flux_process.terminate()
            self.flux_process.wait(timeout=5)
            print("FLUX: Stopped (resources reclaimed for research)")

    def run(self):
        """Main monitoring loop"""
        print("Flux Idle Provider - Monitoring system")
        print("Idle threshold: <20% CPU, <50% RAM")
        print("Reserved for research: P-cores, NPU, GPU, NCS2")
        print("")

        while True:
            idle = self.is_system_idle()

            if idle and not (self.flux_process and self.flux_process.poll() is None):
                print(f"[{time.strftime('%H:%M:%S')}] System idle - starting Flux")
                self.start_flux()

            elif not idle and self.flux_process and self.flux_process.poll() is None:
                print(f"[{time.strftime('%H:%M:%S')}] System active - stopping Flux")
                self.stop_flux()

            time.sleep(self.check_interval)

# CLI
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'daemon':
        provider = FluxIdleProvider()
        provider.run()
    else:
        print("Flux Idle Provider")
        print("Usage:")
        print("  python3 flux_idle_provider.py daemon   # Run monitoring daemon")
        print("")
        print("Features:")
        print("  - Auto-start Flux when system idle")
        print("  - Instant stop when research active")
        print("  - E-cores only (P-cores reserved)")
        print("  - 16GB RAM max (48GB reserved)")
        print("  - NPU/GPU/NCS2 always reserved")
        print("")
        print("Systemd service:")
        print("  sudo systemctl enable flux-idle")
        print("  sudo systemctl start flux-idle")
