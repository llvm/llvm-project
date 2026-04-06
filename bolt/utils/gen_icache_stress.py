#!/usr/bin/env python3
"""
Generate a large C program that stresses instruction cache and CPU BTB.

Strategies used:
1. Many unique functions spread across code to exceed I-cache
2. Unpredictable conditional branches
3. Large function bodies with diverse instruction patterns
4. Three-tier hot/warm/cold code distribution:
   - Function level: 20% hot, 40% warm, 40% cold (never executed)
   - Instruction level: 40% hot, 25% warm, 35% cold (never executed)
5. Optional: Function pointers for indirect branch stress
6. Optional: Computed gotos for additional BTB stress
"""

import random
import argparse


def generate_header(use_func_ptrs, num_hot_funcs, num_warm_funcs, num_cold_funcs):
    code = '''
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Volatile to prevent optimization
volatile uint64_t global_sum = 0;
volatile uint64_t warm_sum = 0;   // Sum from warm code paths
volatile uint64_t cold_sum = 0;   // Sum from cold code paths (should stay 0)

// Prevent inlining to keep functions separate in memory
#define NOINLINE __attribute__((noinline))

'''
    if use_func_ptrs:
        code += '// Function pointer type for indirect calls\n'
        code += 'typedef uint64_t (*compute_func_t)(uint64_t, uint64_t);\n\n'

    # Forward declarations for warm functions
    code += '// Forward declarations for warm (rarely executed) functions\n'
    for i in range(num_warm_funcs):
        code += f'NOINLINE uint64_t warm_func_{i}(uint64_t x, uint64_t seed);\n'
    code += '\n'

    # Forward declarations for cold functions (never executed)
    code += '// Forward declarations for cold (never executed) functions\n'
    for i in range(num_cold_funcs):
        code += f'NOINLINE uint64_t cold_func_{i}(uint64_t x, uint64_t seed);\n'
    code += '\n'

    return code


def generate_hot_block(block_id):
    """Generate a hot code block (main computation)."""
    op = random.choice(['+', '^', '-', '*'])
    const = random.randint(1, 0xFFFF)
    return f'''
        // Hot path {block_id}
        result = result {op} {const}ULL;
        result = (result << 3) | (result >> 61);
        result = result ^ (result >> 17);
        result = result {random.choice(['+', '^'])} 0x{random.randint(1, 0xFFFF):04x}ULL;'''


def generate_warm_block(block_id):
    """Generate a warm code block (rarely executed)."""
    patterns = [
        f'''
        // Warm: rare path {block_id}
        warm_sum += result * 0x{random.randint(1, 0xFFFF):04x}ULL;
        result = result ^ 0x{random.randint(1, 0xFFFFFFFF):08x}ULL;
        result = (result << 7) | (result >> 57);''',

        f'''
        // Warm: edge case {block_id}
        uint64_t tmp_{block_id} = result;
        result = tmp_{block_id} ^ 0x{random.randint(1, 0xFFFF):04x}ULL;
        warm_sum += tmp_{block_id};
        result = result - 0x{random.randint(1, 0xFF):02x}ULL;''',

        f'''
        // Warm: uncommon branch {block_id}
        result = result * 0x{random.randint(3, 17):x}ULL;
        result = result ^ (result >> 11);
        warm_sum += result & 0xFFFFULL;''',
    ]
    return random.choice(patterns)


def generate_cold_block(block_id):
    """Generate a cold code block (never executed)."""
    patterns = [
        f'''
        // Cold: error handler {block_id} (never executed)
        cold_sum += result * 0x{random.randint(1, 0xFFFF):04x}ULL;
        result = result ^ 0x{random.randint(1, 0xFFFFFFFF):08x}ULL;
        result = (result << 11) | (result >> 53);
        cold_sum += result;''',

        f'''
        // Cold: panic path {block_id} (never executed)
        uint64_t err_{block_id} = result;
        cold_sum += err_{block_id};
        result = err_{block_id} ^ 0x{random.randint(1, 0xFFFF):04x}ULL;
        result = result + 0x{random.randint(1, 0xFFF):03x}ULL;''',

        f'''
        // Cold: unreachable {block_id} (never executed)
        result = result * 0x{random.randint(3, 31):x}ULL;
        result = result ^ (result >> 19);
        result = result + 0x{random.randint(1, 0xFFF):03x}ULL;
        cold_sum += result & 0xFFFFULL;''',
    ]
    return random.choice(patterns)


def generate_compute_function_with_tiers(func_id, num_branches=15):
    """Generate a function with 40/25/35 hot/warm/cold code ratio.

    Within each function:
    - 40% hot code (executed most of the time)
    - 25% warm code (rarely executed, ~0.1%)
    - 35% cold code (never executed)
    """
    code = f'''
NOINLINE uint64_t compute_{func_id}(uint64_t x, uint64_t seed) {{
    uint64_t result = x ^ seed ^ {random.randint(1, 0xFFFFFFFF)}ULL;
'''

    # Distribute branches: ~40% hot-only, ~25% hot+warm, ~35% hot+cold
    # Each branch has a hot path that's usually taken
    # Some branches have warm alternatives (rarely taken)
    # Some branches have cold alternatives (never taken)

    block_counter = 0
    for i in range(num_branches):
        # Decide branch type based on 40/25/35 ratio
        roll = random.random()

        if roll < 0.40:
            # Hot-only branch: both paths are hot (always executed)
            code += f'''
    // Branch {i}: hot-only
    if ((result & 1) == 0) {{{generate_hot_block(block_counter)}
    }} else {{{generate_hot_block(block_counter + 1)}
    }}
'''
            block_counter += 2

        elif roll < 0.65:  # 0.40 + 0.25
            # Hot + warm branch: hot path usually taken, warm path rarely
            code += f'''
    // Branch {i}: hot (99.9%) / warm (0.1%)
    if ((result % 1000) != 0) {{{generate_hot_block(block_counter)}
    }} else {{{generate_warm_block(block_counter)}
    }}
'''
            block_counter += 1

        else:  # remaining 0.35
            # Hot + cold branch: hot path always taken, cold path never
            # Use a condition that's always true at runtime
            code += f'''
    // Branch {i}: hot (100%) / cold (0%) - cold path never executed
    if (result != 0xDEADDEADDEADDEADULL) {{{generate_hot_block(block_counter)}
    }} else {{{generate_cold_block(block_counter)}
    }}
'''
            block_counter += 1

    code += '''
    return result;
}
'''
    return code


def generate_warm_function(func_id, num_branches=8):
    """Generate a warm function (rarely executed)."""
    code = f'''
NOINLINE uint64_t warm_func_{func_id}(uint64_t x, uint64_t seed) {{
    // Warm function: rarely executed
    uint64_t result = x ^ seed ^ {random.randint(1, 0xFFFFFFFF)}ULL;
'''

    for i in range(num_branches):
        op = random.choice(['+', '^', '-', '*'])
        const = random.randint(1, 0xFFFF)
        code += f'''
    result = result {op} {const}ULL;
    if ((result & 0xFF) < 128) {{
        result = (result << 5) | (result >> 59);
        warm_sum += result & 0xFFULL;
    }} else {{
        result = result ^ (result >> 13);
        warm_sum += (result >> 8) & 0xFFULL;
    }}
'''

    code += '''
    warm_sum += result;
    return result;
}
'''
    return code


def generate_cold_function(func_id, num_branches=8):
    """Generate a cold function (never executed at runtime)."""
    code = f'''
NOINLINE uint64_t cold_func_{func_id}(uint64_t x, uint64_t seed) {{
    // Cold function: NEVER executed at runtime
    uint64_t result = x ^ seed ^ {random.randint(1, 0xFFFFFFFF)}ULL;
'''

    for i in range(num_branches):
        op = random.choice(['+', '^', '-', '*'])
        const = random.randint(1, 0xFFFF)
        code += f'''
    result = result {op} {const}ULL;
    if ((result & 0xFF) < 128) {{
        result = (result << 5) | (result >> 59);
        cold_sum += result & 0xFFULL;
    }} else {{
        result = result ^ (result >> 13);
        cold_sum += (result >> 8) & 0xFFULL;
    }}
'''

    code += '''
    cold_sum += result;
    return result;
}
'''
    return code


def generate_switch_function(func_id, num_cases=32):
    """Generate a function with a large switch for indirect branch stress."""
    code = f'''
NOINLINE uint64_t switch_func_{func_id}(uint64_t x, uint64_t sel) {{
    uint64_t result = x;
    switch (sel % {num_cases}) {{
'''
    for i in range(num_cases):
        op = random.choice(['+', '^', '-', '*'])
        const = random.randint(1, 0xFFFF)
        code += f'''    case {i}:
        result = result {op} {const}ULL;
        result = (result << {random.randint(1,15)}) | (result >> {64-random.randint(1,15)});
        break;
'''
    code += '''    default:
        result ^= 0xDEADBEEFULL;
    }
    return result;
}
'''
    return code


def generate_goto_function(func_id, num_labels=16):
    """Generate a function with computed gotos for BTB stress."""
    code = f'''
NOINLINE uint64_t goto_func_{func_id}(uint64_t x, uint64_t iterations) {{
    static void* labels[] = {{
'''
    for i in range(num_labels):
        code += f'        &&label_{func_id}_{i},\n'
    code += '''    };

    uint64_t result = x;
    uint64_t count = iterations;

    if (count == 0) return result;

    goto *labels[result % ''' + str(num_labels) + '''];

'''
    for i in range(num_labels):
        op = random.choice(['+', '^', '-'])
        const = random.randint(1, 0xFFFF)
        code += f'''label_{func_id}_{i}:
    result = result {op} {const}ULL;
    count--;
    if (count == 0) return result;
    goto *labels[(result >> {random.randint(1,8)}) % {num_labels}];

'''
    code += '''    return result;
}
'''
    return code


def generate_main(num_hot_funcs, num_warm_funcs, num_cold_funcs,
                  num_switch, num_goto, num_iterations, use_func_ptrs):
    code = ''

    if use_func_ptrs:
        code += f'''
// Function pointer array for indirect calls (hot functions only)
compute_func_t compute_funcs[{num_hot_funcs}];

void init_func_pointers(void) {{
'''
        for i in range(num_hot_funcs):
            code += f'    compute_funcs[{i}] = compute_{i};\n'
        code += '}\n'

    code += f'''
int main(int argc, char** argv) {{
    uint64_t seed = 42;
    if (argc > 1) {{
        seed = (uint64_t)atoll(argv[1]);
    }}
'''

    if use_func_ptrs:
        code += '''
    init_func_pointers();
'''

    code += f'''
    printf("I-cache and BTB stress test (hot/warm/cold distribution)\\n");
    printf("Seed: %lu\\n", seed);
    printf("Iterations: {num_iterations}\\n");
    printf("Functions - Hot: {num_hot_funcs}, Warm: {num_warm_funcs}, Cold: {num_cold_funcs}\\n");

    uint64_t result = seed;

    for (uint64_t iter = 0; iter < {num_iterations}ULL; iter++) {{
'''

    # Call hot compute functions (every iteration)
    code += f'''
        // Hot functions: called every iteration
        switch (iter % {num_hot_funcs}) {{
'''
    for i in range(num_hot_funcs):
        code += f'        case {i}: result = compute_{i}(result, iter); break;\n'
    code += '''        }
'''

    if use_func_ptrs:
        code += f'''
        // Indirect calls via function pointers (hot functions)
        result = compute_funcs[(result ^ iter) % {num_hot_funcs}](result, iter);
'''

    # Call warm functions very rarely
    code += f'''
        // Warm functions: called very rarely (~0.01% of iterations)
        if ((iter % 10000) == 0) {{
            switch ((result >> 4) % {num_warm_funcs}) {{
'''
    for i in range(num_warm_funcs):
        code += f'            case {i}: result = warm_func_{i}(result, iter); break;\n'
    code += '''            }
        }
'''

    # Cold functions are NEVER called - they just exist in the binary
    code += '''
        // Cold functions: NEVER called (exist in binary but unreachable)
        // if (0) { cold_func_0(result, iter); }  // dead code reference
'''

    if num_switch > 0:
        code += f'''
        // Switch-based dispatch
        switch ((iter >> 4) % {num_switch}) {{
'''
        for i in range(num_switch):
            code += f'        case {i}: result = switch_func_{i}(result, iter); break;\n'
        code += '''        }
'''

    if num_goto > 0:
        code += f'''
        // Computed goto functions
        switch ((iter >> 8) % {num_goto}) {{
'''
        for i in range(num_goto):
            code += f'        case {i}: result = goto_func_{i}(result, 16); break;\n'
        code += '''        }
'''

    code += '''
        global_sum += result;
    }

    printf("Result checksum: %lu\\n", global_sum);
    printf("Warm path checksum: %lu\\n", warm_sum);
    printf("Cold path checksum: %lu (should be 0)\\n", cold_sum);

    return 0;
}
'''
    return code


def main():
    parser = argparse.ArgumentParser(
        description='Generate a C program that stresses I-cache and BTB')
    parser.add_argument('--num-functions', type=int, default=5000,
                        help='Total number of compute functions (default: 5000)')
    parser.add_argument('--hot-ratio', type=float, default=0.2,
                        help='Ratio of hot functions (default: 0.2 = 20%%)')
    parser.add_argument('--warm-ratio', type=float, default=0.4,
                        help='Ratio of warm functions (default: 0.4 = 40%%)')
    parser.add_argument('--num-switch', type=int, default=50,
                        help='Number of switch functions (default: 50)')
    parser.add_argument('--num-goto', type=int, default=50,
                        help='Number of computed goto functions (default: 50)')
    parser.add_argument('--branches-per-func', type=int, default=15,
                        help='Branches per compute function (default: 15)')
    parser.add_argument('--iterations', type=int, default=150000000,
                        help='Number of iterations (default: 150000000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for code generation reproducibility (default: 42)')
    parser.add_argument('--use-func-ptrs', action='store_true',
                        help='Use function pointers for indirect calls (default: off)')
    parser.add_argument('--no-switch', action='store_true',
                        help='Disable switch functions')
    parser.add_argument('--no-goto', action='store_true',
                        help='Disable computed goto functions')
    parser.add_argument('-o', '--output', type=str, default='icache_stress.c',
                        help='Output filename (default: icache_stress.c)')

    args = parser.parse_args()
    random.seed(args.seed)

    num_switch = 0 if args.no_switch else args.num_switch
    num_goto = 0 if args.no_goto else args.num_goto

    # Calculate hot/warm/cold function split (20/40/40 by default)
    num_hot_funcs = max(1, int(args.num_functions * args.hot_ratio))
    num_warm_funcs = max(1, int(args.num_functions * args.warm_ratio))
    num_cold_funcs = args.num_functions - num_hot_funcs - num_warm_funcs

    print(f"Generating {args.output}...")
    print(f"  Total functions:   {args.num_functions}")
    print(f"  Hot functions:     {num_hot_funcs} ({args.hot_ratio*100:.0f}%)")
    print(f"  Warm functions:    {num_warm_funcs} ({args.warm_ratio*100:.0f}%)")
    print(f"  Cold functions:    {num_cold_funcs} ({(1-args.hot_ratio-args.warm_ratio)*100:.0f}%)")
    print(f"  Switch functions:  {num_switch}")
    print(f"  Goto functions:    {num_goto}")
    print(f"  Iterations:        {args.iterations}")
    print(f"  Function pointers: {'yes' if args.use_func_ptrs else 'no'}")

    with open(args.output, 'w') as f:
        # Header
        f.write(generate_header(args.use_func_ptrs, num_hot_funcs,
                                num_warm_funcs, num_cold_funcs))

        # Forward declarations for hot functions
        f.write("// Forward declarations for hot (frequently executed) functions\n")
        for i in range(num_hot_funcs):
            f.write(f"NOINLINE uint64_t compute_{i}(uint64_t x, uint64_t seed);\n")
        f.write("\n")

        # Build list of all functions to interleave
        all_functions = []
        for i in range(num_hot_funcs):
            all_functions.append(('hot', i))
        for i in range(num_warm_funcs):
            all_functions.append(('warm', i))
        for i in range(num_cold_funcs):
            all_functions.append(('cold', i))

        # Shuffle to interleave hot/warm/cold functions
        random.shuffle(all_functions)

        # Generate interleaved functions
        f.write("// ============ Compute Functions (interleaved hot/warm/cold) ============\n")
        generated_counts = {'hot': 0, 'warm': 0, 'cold': 0}
        for func_type, func_id in all_functions:
            if func_type == 'hot':
                f.write(generate_compute_function_with_tiers(func_id, args.branches_per_func))
            elif func_type == 'warm':
                f.write(generate_warm_function(func_id, args.branches_per_func // 2))
            else:  # cold
                f.write(generate_cold_function(func_id, args.branches_per_func // 2))

            generated_counts[func_type] += 1
            total_generated = sum(generated_counts.values())
            if total_generated % 100 == 0:
                print(f"  Generated {total_generated}/{len(all_functions)} functions "
                      f"(hot: {generated_counts['hot']}, warm: {generated_counts['warm']}, "
                      f"cold: {generated_counts['cold']})")

        # Switch functions
        if num_switch > 0:
            f.write("\n// ============ Switch Functions ============\n")
            for i in range(num_switch):
                f.write(generate_switch_function(i))

        # Computed goto functions
        if num_goto > 0:
            f.write("\n// ============ Computed Goto Functions ============\n")
            for i in range(num_goto):
                f.write(generate_goto_function(i))

        # Main function
        f.write("\n// ============ Main ============\n")
        f.write(generate_main(num_hot_funcs, num_warm_funcs, num_cold_funcs,
                              num_switch, num_goto, args.iterations, args.use_func_ptrs))

    print(f"Done! Generated {args.output}")
    print(f"\nTo compile and run:")
    print(f"  gcc -O2 -o icache_stress {args.output}")
    print(f"  ./icache_stress")
    print(f"\nFor maximum I-cache stress (larger code):")
    print(f"  gcc -O0 -o icache_stress {args.output}")


if __name__ == '__main__':
    main()
