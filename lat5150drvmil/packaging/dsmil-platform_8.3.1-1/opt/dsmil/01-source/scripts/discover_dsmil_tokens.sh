#!/bin/bash
# DSMIL Token Discovery Script
# Focuses on finding the 72 DSMIL device tokens

echo "=== DSMIL Token Discovery ==="
echo "Looking for 72 DSMIL device tokens (6 groups × 12 devices)"
echo ""

# Create results file
RESULTS_FILE="dsmil_token_discovery_$(date +%Y%m%d_%H%M%S).txt"

# Function to check a token
check_token() {
    local token=$1
    local result=$(echo '1786' | sudo -S smbios-token-ctl --get-token=$token 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "$result"
    else
        echo "not_found"
    fi
}

echo "Phase 1: Checking critical unknown tokens" | tee -a "$RESULTS_FILE"
echo "=========================================" | tee -a "$RESULTS_FILE"

# Check the unknown tokens in military range
echo "Military range (0x8012-0x8014):" | tee -a "$RESULTS_FILE"
for token in 0x8012 0x8013 0x8014; do
    value=$(check_token $token)
    echo "  Token $token: value=$value" | tee -a "$RESULTS_FILE"
done

echo "" | tee -a "$RESULTS_FILE"
echo "F100 Series unknown tokens:" | tee -a "$RESULTS_FILE"
for i in 0x100 0x102 0x110 0x112 0x120 0x122 0x130 0x132 0x140 0x142 0x150 0x152; do
    token=$(printf "0xf%03x" $i)
    value=$(check_token $token)
    if [ "$value" != "not_found" ]; then
        echo "  Token $token: value=$value" | tee -a "$RESULTS_FILE"
    fi
done

echo "" | tee -a "$RESULTS_FILE"
echo "Phase 2: Looking for 72 sequential tokens" | tee -a "$RESULTS_FILE"
echo "=========================================" | tee -a "$RESULTS_FILE"

# Look for groups of 12 sequential tokens (6 groups × 12 devices = 72)
# Check various base addresses
for base in 0x400 0x480 0x500 0x1000 0x1100 0x1200 0x1300 0x1400 0x1500 0x4400 0x5400; do
    echo "Checking base 0x$(printf "%04x" $base)..." | tee -a "$RESULTS_FILE"
    
    found_count=0
    active_count=0
    token_list=""
    
    # Check 72 offsets from base
    for offset in $(seq 0 71); do
        token=$((base + offset))
        value=$(check_token $token)
        
        if [ "$value" != "not_found" ]; then
            ((found_count++))
            token_list="$token_list $token"
            
            if [ "$value" = "1" ]; then
                ((active_count++))
            fi
        fi
    done
    
    if [ $found_count -ge 12 ]; then
        echo "  FOUND: $found_count tokens (possible DSMIL range)" | tee -a "$RESULTS_FILE"
        echo "  Active: $active_count tokens" | tee -a "$RESULTS_FILE"
        
        # Show groups of 12
        for group in 0 1 2 3 4 5; do
            group_start=$((base + group * 12))
            group_end=$((group_start + 11))
            
            group_found=0
            for i in $(seq $group_start $group_end); do
                value=$(check_token $i)
                if [ "$value" != "not_found" ]; then
                    ((group_found++))
                fi
            done
            
            if [ $group_found -gt 0 ]; then
                echo "    Group $group (0x$(printf "%04x" $group_start)-0x$(printf "%04x" $group_end)): $group_found tokens" | tee -a "$RESULTS_FILE"
            fi
        done
    fi
done

echo "" | tee -a "$RESULTS_FILE"
echo "Phase 3: Checking token patterns in unknown ranges" | tee -a "$RESULTS_FILE"
echo "===================================================" | tee -a "$RESULTS_FILE"

# Get all unknown tokens and look for patterns
echo '1786' | sudo -S smbios-token-ctl --dump-tokens 2>&1 | grep -B1 "unknown" | grep "Token:" | awk '{print $2}' | while read token_hex; do
    # Convert to decimal
    token_dec=$((token_hex))
    
    # Check if it's part of a sequence of 12
    base=$((token_dec / 12 * 12))
    offset=$((token_dec % 12))
    
    # If it's at the start of a potential group
    if [ $offset -eq 0 ]; then
        # Check if next 11 tokens exist
        sequential_count=1
        for i in $(seq 1 11); do
            next_token=$((base + i))
            value=$(check_token $next_token)
            if [ "$value" != "not_found" ]; then
                ((sequential_count++))
            fi
        done
        
        if [ $sequential_count -eq 12 ]; then
            echo "  Found complete group of 12 at base 0x$(printf "%04x" $base)" | tee -a "$RESULTS_FILE"
        fi
    fi
done

echo "" | tee -a "$RESULTS_FILE"
echo "=== Discovery Complete ===" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"

# Summary
echo ""
echo "Summary of findings:"
grep -E "FOUND:|Found complete group" "$RESULTS_FILE"