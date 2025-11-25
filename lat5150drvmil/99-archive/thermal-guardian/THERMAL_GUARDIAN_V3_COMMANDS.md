# Thermal Guardian v3.0 - Working Commands

## Essential Commands Only

### Test Sensors
```bash
sudo python3 thermal_guardian.py --test-sensors
```

### Run Diagnostics
```bash
sudo python3 thermal_guardian.py --diagnostics
```

### Start Service (Daemon)
```bash
sudo python3 thermal_guardian.py --daemon
```

### Check Status
```bash
sudo python3 thermal_guardian.py --status
```

### Additional Commands

#### View Version
```bash
python3 thermal_guardian.py --version
```

#### Emergency Test (5 seconds)
```bash
sudo python3 thermal_guardian.py --emergency-test
```

#### View Metrics
```bash
sudo python3 thermal_guardian.py --metrics
```

#### Run in Foreground (for debugging)
```bash
sudo python3 thermal_guardian.py
```

## Quick Start

1. **Test sensors first:**
   ```bash
   sudo python3 thermal_guardian.py --test-sensors
   ```

2. **Check diagnostics:**
   ```bash
   sudo python3 thermal_guardian.py --diagnostics
   ```

3. **Start daemon:**
   ```bash
   sudo python3 thermal_guardian.py --daemon
   ```

4. **Monitor status:**
   ```bash
   sudo python3 thermal_guardian.py --status
   ```

## Notes
- All commands use `thermal_guardian.py` (v3.0 Enhanced Edition)
- Root (sudo) required for hardware access
- Old test scripts and v2.0 files can be archived/removed