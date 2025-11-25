#!/usr/bin/env python3
"""GNA Presence Detector - Hardware-based user detection"""
from openvino import Core
import time

class GNAPresenceDetector:
    def __init__(self):
        self.core = Core()
        self.last_input_time = time.time()
        self.presence_state = 'ACTIVE'

    def detect_presence(self):
        """
        Detect user presence via input patterns
        Returns: ACTIVE, IDLE, or AWAY
        """
        idle_sec = time.time() - self.last_input_time

        if idle_sec < 60:
            return 'ACTIVE'
        elif idle_sec < 900:  # 15 min
            return 'IDLE'
        else:
            return 'AWAY'

    def get_flux_allocation(self):
        """Get Flux resource allocation based on presence"""
        presence = self.detect_presence()

        allocations = {
            'ACTIVE': {
                'cores': '20-21',  # LP E-cores only
                'threads': 2,
                'ram_gb': 4,
                'earnings_month': 20
            },
            'IDLE': {
                'cores': '12-21',  # E-cores + LP E-cores
                'threads': 10,
                'ram_gb': 16,
                'earnings_month': 100
            },
            'AWAY': {
                'cores': '0-21',  # All cores (reserves NPU/GPU)
                'threads': 22,
                'ram_gb': 48,
                'earnings_month': 200
            }
        }

        return allocations[presence]

    def update_activity(self):
        """Call when user input detected"""
        self.last_input_time = time.time()

# CLI
if __name__ == "__main__":
    detector = GNAPresenceDetector()

    print("GNA Presence Detector")
    print("=" * 40)

    for _ in range(5):
        presence = detector.detect_presence()
        alloc = detector.get_flux_allocation()

        print(f"\nPresence: {presence}")
        print(f"Flux cores: {alloc['cores']}")
        print(f"Flux earnings: ${alloc['earnings_month']}/month")

        time.sleep(2)
