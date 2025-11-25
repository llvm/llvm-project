#!/usr/bin/env python3
"""
DSMIL Token Pattern Analyzer
Analyzes SMBIOS tokens to identify potential DSMIL device control tokens
"""

import subprocess
import re
from collections import defaultdict

def get_all_tokens():
    """Get all SMBIOS tokens with their states"""
    tokens = {}
    try:
        # Run smbios-token-ctl with sudo
        result = subprocess.run(
            ['sudo', '-S', 'smbios-token-ctl', '--dump-tokens'],
            input=b'1786\n',
            capture_output=True,
            text=True
        )
        
        lines = result.stdout.split('\n')
        current_token = None
        
        for line in lines:
            # Parse token line
            token_match = re.search(r'Token:\s+(0x[0-9a-fA-F]+)\s+-\s+(.+)', line)
            if token_match:
                current_token = token_match.group(1)
                name = token_match.group(2)
                tokens[current_token] = {'name': name, 'value': None}
            
            # Parse value line
            elif current_token and 'value:' in line:
                value_match = re.search(r'value:\s+bool\s+=\s+(true|false)', line)
                if value_match:
                    tokens[current_token]['value'] = value_match.group(1) == 'true'
    
    except Exception as e:
        print(f"Error getting tokens: {e}")
    
    return tokens

def analyze_patterns(tokens):
    """Analyze tokens for DSMIL patterns"""
    print("=== DSMIL Token Pattern Analysis ===\n")
    
    # Group tokens by range
    ranges = defaultdict(list)
    for token_hex, info in tokens.items():
        token_val = int(token_hex, 16)
        
        # Categorize by range
        if 0x8000 <= token_val <= 0x8014:
            ranges['Military (0x8000-0x8014)'].append((token_hex, info))
        elif 0xF100 <= token_val <= 0xF152:
            ranges['F100 Series'].append((token_hex, info))
        elif 0xF600 <= token_val <= 0xF601:
            ranges['Security (0xF600-0xF601)'].append((token_hex, info))
        elif 0x1000 <= token_val <= 0x1FFF:
            ranges['1000 Series'].append((token_hex, info))
        elif 0x2000 <= token_val <= 0x2FFF:
            ranges['2000 Series'].append((token_hex, info))
        elif 0x3000 <= token_val <= 0x3FFF:
            ranges['3000 Series'].append((token_hex, info))
        elif 0x4000 <= token_val <= 0x4FFF:
            ranges['4000 Series'].append((token_hex, info))
        elif 0x5000 <= token_val <= 0x5FFF:
            ranges['5000 Series'].append((token_hex, info))
        elif 0x6000 <= token_val <= 0x6FFF:
            ranges['6000 Series'].append((token_hex, info))
        elif 0x7000 <= token_val <= 0x7FFF:
            ranges['7000 Series'].append((token_hex, info))
    
    # Print analysis
    print("Token Distribution by Range:")
    print("-" * 50)
    for range_name, token_list in sorted(ranges.items()):
        print(f"\n{range_name}: {len(token_list)} tokens")
        
        # Show first few tokens in each range
        unknown_count = sum(1 for _, info in token_list if 'unknown' in info['name'].lower())
        active_count = sum(1 for _, info in token_list if info['value'] == True)
        
        print(f"  Unknown tokens: {unknown_count}")
        print(f"  Active tokens: {active_count}")
        
        if unknown_count > 0:
            print("  Unknown token details:")
            for token_hex, info in token_list:
                if 'unknown' in info['name'].lower():
                    print(f"    {token_hex}: value={info['value']}")
    
    # Look for sequential patterns (72 devices = 6 groups × 12 devices)
    print("\n" + "=" * 50)
    print("Looking for 72-device patterns...")
    print("-" * 50)
    
    # Check for sequential tokens in groups of 12
    for base in [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000, 0x8000, 0xF100]:
        sequential = []
        for offset in range(72):  # Check for 72 sequential tokens
            token_hex = f"0x{base + offset:04x}"
            if token_hex in tokens:
                sequential.append(token_hex)
        
        if len(sequential) >= 12:  # At least one group worth
            print(f"\nFound {len(sequential)} sequential tokens starting at 0x{base:04x}:")
            print(f"  Range: {sequential[0]} to {sequential[-1]}")
            
            # Group by 12 (device group size)
            for group_idx in range(0, len(sequential), 12):
                group_tokens = sequential[group_idx:group_idx+12]
                if len(group_tokens) == 12:
                    active = sum(1 for t in group_tokens if tokens[t]['value'] == True)
                    unknown = sum(1 for t in group_tokens if 'unknown' in tokens[t]['name'].lower())
                    print(f"    Group {group_idx//12}: {len(group_tokens)} tokens, {active} active, {unknown} unknown")

def check_specific_ranges():
    """Check specific token ranges that might be DSMIL"""
    print("\n" + "=" * 50)
    print("Checking Specific DSMIL Candidate Ranges")
    print("-" * 50)
    
    # Ranges to check based on agent analysis
    ranges_to_check = [
        (0x8012, 0x8014, "Military Unknown Range"),
        (0xF100, 0xF152, "F100 Series (Unknown)"),
        (0xF600, 0xF601, "Security Range"),
        (0x4400, 0x447F, "Hypothetical DSMIL Range"),
        (0x5400, 0x547F, "Alternative DSMIL Range"),
    ]
    
    for start, end, description in ranges_to_check:
        print(f"\n{description} (0x{start:04x} - 0x{end:04x}):")
        try:
            found = 0
            active = 0
            for token_val in range(start, end + 1):
                token_hex = f"0x{token_val:04x}"
                result = subprocess.run(
                    ['sudo', '-S', 'smbios-token-ctl', '--get-token', str(token_val)],
                    input=b'1786\n',
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0:
                    found += 1
                    value = result.stdout.strip()
                    if value == '1':
                        active += 1
                        print(f"  {token_hex}: ACTIVE")
                    else:
                        print(f"  {token_hex}: inactive")
            
            if found > 0:
                print(f"  Summary: {found} tokens found, {active} active")
            else:
                print(f"  No tokens found in this range")
        except Exception as e:
            print(f"  Error checking range: {e}")

def main():
    print("DSMIL Token Pattern Analyzer")
    print("=" * 50)
    
    # Get all tokens
    print("Retrieving all SMBIOS tokens...")
    tokens = get_all_tokens()
    print(f"Found {len(tokens)} total tokens\n")
    
    # Analyze patterns
    analyze_patterns(tokens)
    
    # Check specific ranges
    check_specific_ranges()
    
    # Summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("-" * 50)
    print(f"Total tokens analyzed: {len(tokens)}")
    
    # Count unknown tokens
    unknown_tokens = [(t, info) for t, info in tokens.items() if 'unknown' in info['name'].lower()]
    print(f"Unknown tokens: {len(unknown_tokens)}")
    
    # Show critical unknown tokens
    critical_unknowns = [t for t, _ in unknown_tokens if 
                         int(t, 16) in range(0x8012, 0x8015) or 
                         int(t, 16) in range(0xF100, 0xF153) or
                         int(t, 16) in range(0xF600, 0xF602)]
    
    if critical_unknowns:
        print(f"\nCritical unknown tokens in target ranges: {len(critical_unknowns)}")
        for token in critical_unknowns:
            info = tokens[token]
            print(f"  {token}: value={info['value']} ({info['name']})")
    
    print("\nRecommendation: Focus on unknown tokens in 0x8012-0x8014 and 0xF100-0xF152 ranges")
    print("These are most likely to be DSMIL control tokens based on:")
    print("  1. They are marked as 'unknown' (undocumented)")
    print("  2. They are in military/security token ranges")
    print("  3. Token 0x8013 is currently active (true)")

if __name__ == "__main__":
    main()