#!/bin/bash
# WMI-based control for locked SMBIOS tokens
# This script uses Dell WMI Sysman to control hardware

WMI_BASE="/sys/class/firmware-attributes/dell-wmi-sysman"

control_via_wmi() {
    local attribute=$1
    local value=$2
    
    echo "Setting $attribute to $value via WMI..."
    
    # Check if attribute exists
    if [ ! -d "$WMI_BASE/attributes/$attribute" ]; then
        echo "Attribute $attribute not found"
        return 1
    fi
    
    # Read current value
    current=$(cat "$WMI_BASE/attributes/$attribute/current_value" 2>/dev/null)
    echo "Current value: $current"
    
    # Set new value (requires authentication)
    echo "$value" | sudo tee "$WMI_BASE/attributes/$attribute/current_value" > /dev/null
    
    # Verify change
    new_value=$(cat "$WMI_BASE/attributes/$attribute/current_value" 2>/dev/null)
    echo "New value: $new_value"
    
    if [ "$new_value" = "$value" ]; then
        echo "✅ Successfully changed"
        return 0
    else
        echo "❌ Change failed (may require password authentication)"
        return 1
    fi
}

# Example: Control power management (may affect token 0x0480)
control_via_wmi "PowerManagement" "Custom"

# Example: Control thermal management (may affect token 0x0489)
control_via_wmi "ThermalManagement" "Performance"

# List all available attributes
echo ""
echo "Available WMI attributes:"
ls -1 "$WMI_BASE/attributes/" | head -20
