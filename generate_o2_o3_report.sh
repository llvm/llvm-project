#!/bin/bash

# O2 vs O3 Pass Comparison Report Generator
# Generates a detailed report comparing optimization passes between O2 and O3

REPORT_FILE="O2_vs_O3_comparison_report.txt"
TEST_FILE="test.c"

echo "Generating O2 vs O3 Pass Comparison Report..."

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "Warning: $TEST_FILE not found. Creating a sample test file..."
    cat > "$TEST_FILE" << 'EOF'
int main() {
  int a = 5;
  int b = 10;
  int c = a + b;

  // Dead code that should be eliminated
  int dead1 = 100;
  int dead2 = 200;
  int dead3 = dead1 + dead2; // This is never used

  // Redundant calculations
  int x = a + b;  // Same as c
  int y = 5 + 10; // Constant folding opportunity

  // Unused loop
  for (int i = 0; i < 10; i++) {
    int temp = i * 2; // Dead code in loop
  }

  return c;
}
EOF
fi

# Generate pass lists for O2 and O3
echo "Collecting O2 pass information..."
./bin/clang -O2 -mllvm -debug-pass=List "$TEST_FILE" 2>&1 | grep "^Running pass:" > o2_passes_raw.txt

echo "Collecting O3 pass information..."
./bin/clang -O3 -mllvm -debug-pass=List "$TEST_FILE" 2>&1 | grep "^Running pass:" > o3_passes_raw.txt

# Clean the pass names (remove "Running pass: " prefix)
sed 's/Running pass: //' o2_passes_raw.txt | sort > o2_passes_clean.txt
sed 's/Running pass: //' o3_passes_raw.txt | sort > o3_passes_clean.txt

# Start generating the report
cat > "$REPORT_FILE" << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                     LLVM O2 vs O3 OPTIMIZATION COMPARISON                   ║
║                              Pass Analysis Report                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF

# Add metadata
echo "Report Generated: $(date)" >> "$REPORT_FILE"
echo "LLVM Version: $(./bin/clang --version | head -1)" >> "$REPORT_FILE"
echo "Test File: $TEST_FILE" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Calculate statistics
O2_COUNT=$(wc -l < o2_passes_raw.txt)
O3_COUNT=$(wc -l < o3_passes_raw.txt)
COMMON_COUNT=$(comm -12 o2_passes_clean.txt o3_passes_clean.txt | wc -l)
O2_ONLY_COUNT=$(comm -23 o2_passes_clean.txt o3_passes_clean.txt | wc -l)
O3_ONLY_COUNT=$(comm -13 o2_passes_clean.txt o3_passes_clean.txt | wc -l)

# Add executive summary
cat >> "$REPORT_FILE" << EOF
═══════════════════════════════════════════════════════════════════════════════
                              EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════════════

• O2 Optimization Level: $O2_COUNT total passes
• O3 Optimization Level: $O3_COUNT total passes
• Additional passes in O3: $(($O3_COUNT - $O2_COUNT))
• Common passes: $COMMON_COUNT
• O2-only passes: $O2_ONLY_COUNT
• O3-only passes: $O3_ONLY_COUNT

Key Finding: O3 adds $(echo "scale=1; (($O3_COUNT - $O2_COUNT) * 100.0) / $O2_COUNT" | bc -l)% more optimization passes compared to O2

EOF

# Detailed analysis section
cat >> "$REPORT_FILE" << 'EOF'

═══════════════════════════════════════════════════════════════════════════════
                              DETAILED ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

EOF

# O3-only passes (what's NEW in O3)
echo "┌─ PASSES UNIQUE TO O3 (Additional Optimizations) ──────────────────────────┐" >> "$REPORT_FILE"
echo "│ These passes run only at O3 level, providing more aggressive optimization  │" >> "$REPORT_FILE"
echo "│                                                                            │" >> "$REPORT_FILE"

if [ $O3_ONLY_COUNT -gt 0 ]; then
    comm -13 o2_passes_clean.txt o3_passes_clean.txt | nl -nln | sed 's/^/│ /' >> "$REPORT_FILE"
else
    echo "│ No O3-specific passes found                                                │" >> "$REPORT_FILE"
fi

echo "└────────────────────────────────────────────────────────────────────────────┘" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# O2-only passes (if any)
if [ $O2_ONLY_COUNT -gt 0 ]; then
    echo "┌─ PASSES UNIQUE TO O2 (Not in O3) ─────────────────────────────────────────┐" >> "$REPORT_FILE"
    echo "│ These passes run in O2 but not O3 (unusual, may indicate pass reordering) │" >> "$REPORT_FILE"
    echo "│                                                                            │" >> "$REPORT_FILE"
    comm -23 o2_passes_clean.txt o3_passes_clean.txt | nl -nln | sed 's/^/│ /' >> "$REPORT_FILE"
    echo "└────────────────────────────────────────────────────────────────────────────┘" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

# Pass frequency analysis
echo "┌─ PASS FREQUENCY ANALYSIS ─────────────────────────────────────────────────┐" >> "$REPORT_FILE"
echo "│ How many times each pass runs (some passes may run multiple times)        │" >> "$REPORT_FILE"
echo "│                                                                            │" >> "$REPORT_FILE"

# Count pass frequencies for O2
echo "│ O2 Pass Frequencies:                                                       │" >> "$REPORT_FILE"
sed 's/Running pass: //' o2_passes_raw.txt | sort | uniq -c | sort -nr | head -10 | \
    awk '{printf "│   %-50s %s times\n", $2, $1}' >> "$REPORT_FILE"

echo "│                                                                            │" >> "$REPORT_FILE"
echo "│ O3 Pass Frequencies:                                                       │" >> "$REPORT_FILE"
sed 's/Running pass: //' o3_passes_raw.txt | sort | uniq -c | sort -nr | head -10 | \
    awk '{printf "│   %-50s %s times\n", $2, $1}' >> "$REPORT_FILE"

echo "└────────────────────────────────────────────────────────────────────────────┘" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Categorize passes by type
echo "┌─ PASS CATEGORIZATION (O3-only passes) ────────────────────────────────────┐" >> "$REPORT_FILE"
echo "│ Categorizing O3-specific passes by optimization type                       │" >> "$REPORT_FILE"
echo "│                                                                            │" >> "$REPORT_FILE"

# Function to categorize passes
categorize_passes() {
    local file=$1
    local prefix=$2
    
    # Loop optimizations
    local loop_passes=$(grep -i "loop\|unroll\|vectoriz\|licm" "$file" || true)
    if [ -n "$loop_passes" ]; then
        echo "${prefix} Loop Optimizations:" >> "$REPORT_FILE"
        echo "$loop_passes" | sed "s/^/${prefix}   /" >> "$REPORT_FILE"
        echo "${prefix}" >> "$REPORT_FILE"
    fi
    
    # Function optimizations
    local function_passes=$(grep -i "inline\|function\|call" "$file" || true)
    if [ -n "$function_passes" ]; then
        echo "${prefix} Function/Inlining Optimizations:" >> "$REPORT_FILE"
        echo "$function_passes" | sed "s/^/${prefix}   /" >> "$REPORT_FILE"
        echo "${prefix}" >> "$REPORT_FILE"
    fi
    
    # Memory optimizations
    local memory_passes=$(grep -i "mem\|alias\|gvn\|load\|store" "$file" || true)
    if [ -n "$memory_passes" ]; then
        echo "${prefix} Memory Optimizations:" >> "$REPORT_FILE"
        echo "$memory_passes" | sed "s/^/${prefix}   /" >> "$REPORT_FILE"
        echo "${prefix}" >> "$REPORT_FILE"
    fi
    
    # Dead code elimination
    local dce_passes=$(grep -i "dce\|dead\|unused\|eliminate" "$file" || true)
    if [ -n "$dce_passes" ]; then
        echo "${prefix} Dead Code Elimination:" >> "$REPORT_FILE"
        echo "$dce_passes" | sed "s/^/${prefix}   /" >> "$REPORT_FILE"
        echo "${prefix}" >> "$REPORT_FILE"
    fi
}

# Create temporary file with O3-only passes
comm -13 o2_passes_clean.txt o3_passes_clean.txt > o3_only_passes.txt
categorize_passes "o3_only_passes.txt" "│"

echo "└────────────────────────────────────────────────────────────────────────────┘" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Complete pass execution order comparison
echo "┌─ COMPLETE PASS EXECUTION ORDER ───────────────────────────────────────────┐" >> "$REPORT_FILE"
echo "│ Side-by-side comparison of pass execution order                            │" >> "$REPORT_FILE"
echo "│                                                                            │" >> "$REPORT_FILE"

echo "│ O2 Passes (in execution order):                                            │" >> "$REPORT_FILE"
echo "│ ════════════════════════════════                                           │" >> "$REPORT_FILE"
sed 's/Running pass: //' o2_passes_raw.txt | nl -nln | sed 's/^/│ /' >> "$REPORT_FILE"

echo "│                                                                            │" >> "$REPORT_FILE"
echo "│ O3 Passes (in execution order):                                            │" >> "$REPORT_FILE"
echo "│ ════════════════════════════════                                           │" >> "$REPORT_FILE"
sed 's/Running pass: //' o3_passes_raw.txt | nl -nln | sed 's/^/│ /' >> "$REPORT_FILE"

echo "└────────────────────────────────────────────────────────────────────────────┘" >> "$REPORT_FILE"

# Performance implications
cat >> "$REPORT_FILE" << 'EOF'

═══════════════════════════════════════════════════════════════════════════════
                           PERFORMANCE IMPLICATIONS
═══════════════════════════════════════════════════════════════════════════════

Based on the pass analysis:

COMPILATION TIME:
• O3 requires more compilation time due to additional optimization passes
• The extra passes in O3 perform more complex analysis and transformations

RUNTIME PERFORMANCE:
• O3's additional passes typically result in better runtime performance
• Common O3 improvements include:
  - More aggressive function inlining
  - Better loop optimizations (unrolling, vectorization)
  - More sophisticated dead code elimination
  - Enhanced constant propagation and folding

TRADE-OFFS:
• O2: Balanced optimization (good performance, reasonable compile time)
• O3: Maximum optimization (best performance, longer compile time)

EOF

# Add recommendations
cat >> "$REPORT_FILE" << 'EOF'

═══════════════════════════════════════════════════════════════════════════════
                              RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════════════

FOR DEVELOPMENT:
• Use O2 for faster compilation during development cycles
• Switch to O3 for production builds where runtime performance is critical

FOR PRODUCTION:
• Use O3 when maximum runtime performance is required
• Consider O2 if compilation time is a constraint in CI/CD pipelines

FOR DEBUGGING:
• Use this report to understand which optimizations are applied
• Compare with -O0 and -O1 to see the full optimization progression

EOF

# Footer
cat >> "$REPORT_FILE" << 'EOF'

═══════════════════════════════════════════════════════════════════════════════
                                 END OF REPORT
═══════════════════════════════════════════════════════════════════════════════

Generated by LLVM Pass Analysis Tool
EOF

# Cleanup temporary files
rm -f o2_passes_raw.txt o3_passes_raw.txt o2_passes_clean.txt o3_passes_clean.txt o3_only_passes.txt

# Report completion
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    REPORT GENERATED SUCCESSFULLY              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Report saved to: $REPORT_FILE"
echo ""
echo "Summary:"
echo "  • O2 passes: $O2_COUNT"
echo "  • O3 passes: $O3_COUNT"
echo "  • Additional in O3: $(($O3_COUNT - $O2_COUNT))"
echo "  • O3-only passes: $O3_ONLY_COUNT"
echo ""
echo "To view the report:"
echo "  cat $REPORT_FILE"
echo "  # or"
echo "  less $REPORT_FILE"