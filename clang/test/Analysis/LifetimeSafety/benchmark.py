import sys
import argparse
import subprocess
import tempfile
import json
import os
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t


def generate_cpp_cycle_test(n: int) -> str:
    """
    Generates a C++ code snippet with a specified number of pointers in a cycle.
    Creates a while loop that rotates N pointers.
    This pattern tests the convergence speed of the dataflow analysis when
    reaching its fixed point.

    Example:
        struct MyObj { int id; ~MyObj() {} };

        void long_cycle_4(bool condition) {
            MyObj v1{1};
            MyObj v2{1};
            MyObj v3{1};
            MyObj v4{1};

            MyObj* p1 = &v1;
            MyObj* p2 = &v2;
            MyObj* p3 = &v3;
            MyObj* p4 = &v4;

            while (condition) {
                MyObj* temp = p1;
                p1 = p2;
                p2 = p3;
                p3 = p4;
                p4 = temp;
            }
        }
    """
    if n <= 0:
        return "// Number of variables must be positive."

    cpp_code = "struct MyObj { int id; ~MyObj() {} };\n\n"
    cpp_code += f"void long_cycle_{n}(bool condition) {{\n"
    for i in range(1, n + 1):
        cpp_code += f"  MyObj v{i}{{1}};\n"
    cpp_code += "\n"
    for i in range(1, n + 1):
        cpp_code += f"  MyObj* p{i} = &v{i};\n"

    cpp_code += "\n  while (condition) {\n"
    if n > 0:
        cpp_code += f"    MyObj* temp = p1;\n"
        for i in range(1, n):
            cpp_code += f"    p{i} = p{i+1};\n"
        cpp_code += f"    p{n} = temp;\n"
    cpp_code += "  }\n}\n"
    cpp_code += f"\nint main() {{ long_cycle_{n}(false); return 0; }}\n"
    return cpp_code


def generate_cpp_merge_test(n: int) -> str:
    """
    Creates N independent if statements that merge at a single point.
    This pattern specifically stresses the performance of the
    'LifetimeLattice::join' operation.

    Example:
        struct MyObj { int id; ~MyObj() {} };

        void conditional_merges_4(bool condition) {
            MyObj v1, v2, v3, v4;
            MyObj *p1 = nullptr, *p2 = nullptr, *p3 = nullptr, *p4 = nullptr;

            if(condition) { p1 = &v1; }
            if(condition) { p2 = &v2; }
            if(condition) { p3 = &v3; }
            if(condition) { p4 = &v4; }
        }
    """
    if n <= 0:
        return "// Number of variables must be positive."

    cpp_code = "struct MyObj { int id; ~MyObj() {} };\n\n"
    cpp_code += f"void conditional_merges_{n}(bool condition) {{\n"
    decls = [f"v{i}" for i in range(1, n + 1)]
    cpp_code += f"  MyObj {', '.join(decls)};\n"
    ptr_decls = [f"*p{i} = nullptr" for i in range(1, n + 1)]
    cpp_code += f"  MyObj {', '.join(ptr_decls)};\n\n"

    for i in range(1, n + 1):
        cpp_code += f"  if(condition) {{ p{i} = &v{i}; }}\n"

    cpp_code += "}\n"
    cpp_code += f"\nint main() {{ conditional_merges_{n}(false); return 0; }}\n"
    return cpp_code


def generate_cpp_nested_loop_test(n: int) -> str:
    """
    Generates C++ code with N levels of nested loops.
    This pattern tests how analysis performance scales with loop nesting depth,
    which is a key factor in the complexity of dataflow analyses on structured
    control flow.

    Example (n=3):
        struct MyObj { int id; ~MyObj() {} };
        void nested_loops_3() {
            MyObj* p = nullptr;
            for(int i0=0; i0<2; ++i0) {
                MyObj s0;
                p = &s0;
                for(int i1=0; i1<2; ++i1) {
                    MyObj s1;
                    p = &s1;
                    for(int i2=0; i2<2; ++i2) {
                        MyObj s2;
                        p = &s2;
                    }
                }
            }
        }
    """
    if n <= 0:
        return "// Nesting depth must be positive."

    cpp_code = "struct MyObj { int id; ~MyObj() {} };\n\n"
    cpp_code += f"void nested_loops_{n}() {{\n"
    cpp_code += "    MyObj* p = nullptr;\n"

    for i in range(n):
        indent = "    " * (i + 1)
        cpp_code += f"{indent}for(int i{i}=0; i{i}<2; ++i{i}) {{\n"
        cpp_code += f"{indent}    MyObj s{i}; p = &s{i};\n"

    for i in range(n - 1, -1, -1):
        indent = "    " * (i + 1)
        cpp_code += f"{indent}}}\n"

    cpp_code += "}\n"
    cpp_code += f"\nint main() {{ nested_loops_{n}(); return 0; }}\n"
    return cpp_code


def generate_cpp_switch_fan_out_test(n: int) -> str:
    """
    Generates C++ code with a switch statement with N branches.
    Each branch 'i' 'uses' (reads) a single, unique pointer 'pi'.
    This pattern creates a "fan-in" join point for the backward
    liveness analysis, stressing the LivenessMap::join operation
    by forcing it to merge N disjoint, single-element sets of live origins.
    The resulting complexity for LiveOrigins should be O(n log n) or higher.

    Example (n=3):
        struct MyObj { int id; ~MyObj() {} };

        void switch_fan_out_3(int condition) {
            MyObj v1{1}; MyObj v2{1}; MyObj v3{1};
            MyObj* p1 = &v1; MyObj* p2 = &v2; MyObj* p3 = &v3;

            switch (condition % 3) {
                case 0:
                    p1->id = 1;
                    break;
                case 1:
                    p2->id = 1;
                    break;
                case 2:
                    p3->id = 1;
                    break;
            }
        }
    """
    if n <= 0:
        return "// Number of variables must be positive."

    cpp_code = "struct MyObj { int id; ~MyObj() {} };\n\n"
    cpp_code += f"void switch_fan_out{n}(int condition) {{\n"
    # Generate N distinct objects
    for i in range(1, n + 1):
        cpp_code += f"  MyObj v{i}{{1}};\n"
    cpp_code += "\n"
    # Generate N distinct pointers, each as a separate variable
    for i in range(1, n + 1):
        cpp_code += f"  MyObj* p{i} = &v{i};\n"
    cpp_code += "\n"

    cpp_code += f"  switch (condition % {n}) {{\n"
    for case_num in range(n):
        cpp_code += f"    case {case_num}:\n"
        cpp_code += f"      p{case_num + 1}->id = 1;\n"
        cpp_code += "      break;\n"

    cpp_code += "  }\n}\n"
    cpp_code += f"\nint main() {{ switch_fan_out{n}(0); return 0; }}\n"
    return cpp_code


def analyze_trace_file(trace_path: str) -> dict:
    """
    Parses the -ftime-trace JSON output to find durations for the lifetime
    analysis and its sub-phases.
    Returns a dictionary of durations in microseconds.
    """
    durations = {
        "lifetime_us": 0.0,
        "total_us": 0.0,
        "fact_gen_us": 0.0,
        "loan_prop_us": 0.0,
        "live_origins_us": 0.0,
    }
    event_name_map = {
        "LifetimeSafetyAnalysis": "lifetime_us",
        "ExecuteCompiler": "total_us",
        "FactGenerator": "fact_gen_us",
        "LoanPropagation": "loan_prop_us",
        "LiveOrigins": "live_origins_us",
    }
    try:
        with open(trace_path, "r") as f:
            trace_data = json.load(f)
            for event in trace_data.get("traceEvents", []):
                event_name = event.get("name")
                if event_name in event_name_map:
                    key = event_name_map[event_name]
                    durations[key] += float(event.get("dur", 0))
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing trace file {trace_path}: {e}", file=sys.stderr)
        return {key: 0.0 for key in durations}
    return durations


def power_law(n, c, k):
    """Represents the power law function: y = c * n^k"""
    return c * np.power(n, k)


def human_readable_time(ms: float) -> str:
    """Converts milliseconds to a human-readable string (ms or s)."""
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.2f} ms"


def calculate_complexity(n_data, y_data) -> tuple[float | None, float | None]:
    """
    Calculates the exponent 'k' for the power law fit y = c * n^k.
    Returns a tuple of (k, k_standard_error).
    """
    try:
        if len(n_data) < 3 or np.all(y_data < 1e-6) or np.var(y_data) < 1e-6:
            return None, None

        non_zero_indices = y_data > 0
        if np.sum(non_zero_indices) < 3:
            return None, None

        n_fit, y_fit = n_data[non_zero_indices], y_data[non_zero_indices]
        popt, pcov = curve_fit(power_law, n_fit, y_fit, p0=[0, 1], maxfev=5000)
        k_stderr = np.sqrt(np.diag(pcov))[1]
        return popt[1], k_stderr
    except (RuntimeError, ValueError):
        return None, None


def generate_markdown_report(results: dict) -> str:
    """Generates a concise, Markdown-formatted report from the benchmark results."""
    report = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    report.append(f"# Lifetime Analysis Performance Report")
    report.append(f"> Generated on: {timestamp}")
    report.append("\n---\n")

    for test_name, data in results.items():
        title = data["title"]
        report.append(f"## Test Case: {title}")
        report.append("\n**Timing Results:**\n")

        # Table header
        report.append(
            "| N (Input Size) | Total Time | Analysis Time (%) | Fact Generator (%) | Loan Propagation (%) | Live Origins (%) |"
        )
        report.append(
            "|:---------------|-----------:|------------------:|-------------------:|---------------------:|------------------:|"
        )

        # Table rows
        n_data = np.array(data["n"])
        total_ms_data = np.array(data["total_ms"])
        for i in range(len(n_data)):
            total_t = total_ms_data[i]
            if total_t < 1e-6:
                total_t = 1.0  # Avoid division by zero

            row = [
                f"| {n_data[i]:<14} |",
                f"{human_readable_time(total_t):>10} |",
                f"{data['lifetime_ms'][i] / total_t * 100:>17.2f}% |",
                f"{data['fact_gen_ms'][i] / total_t * 100:>18.2f}% |",
                f"{data['loan_prop_ms'][i] / total_t * 100:>20.2f}% |",
                f"{data['live_origins_ms'][i] / total_t * 100:>17.2f}% |",
            ]
            report.append(" ".join(row))

        report.append("\n**Complexity Analysis:**\n")
        report.append("| Analysis Phase    | Complexity O(n<sup>k</sup>) | ")
        report.append("|:------------------|:--------------------------|")

        analysis_phases = {
            "Total Analysis": data["lifetime_ms"],
            "FactGenerator": data["fact_gen_ms"],
            "LoanPropagation": data["loan_prop_ms"],
            "LiveOrigins": data["live_origins_ms"],
        }

        for phase_name, y_data in analysis_phases.items():
            k, delta = calculate_complexity(n_data, np.array(y_data))
            if k is not None and delta is not None:
                complexity_str = f"O(n<sup>{k:.2f}</sup> &pm; {delta:.2f})"
            else:
                complexity_str = "(Negligible)"
            report.append(f"| {phase_name:<17} | {complexity_str:<25} |")

        report.append("\n---\n")

    return "\n".join(report)


def run_single_test(
    clang_binary: str, output_dir: str, test_name: str, generator_func, n: int
) -> dict:
    """Generates, compiles, and benchmarks a single test case."""
    print(f"--- Running Test: {test_name.capitalize()} with N={n} ---")

    generated_code = generator_func(n)

    base_name = f"test_{test_name}_{n}"
    source_file = os.path.join(output_dir, f"{base_name}.cpp")
    trace_file = os.path.join(output_dir, f"{base_name}.json")

    with open(source_file, "w") as f:
        f.write(generated_code)

    clang_command = [
        clang_binary,
        "-c",
        "-o",
        "/dev/null",
        "-ftime-trace=" + trace_file,
        "-Xclang",
        "-fexperimental-lifetime-safety",
        "-std=c++17",
        source_file,
    ]

    try:
        result = subprocess.run(
            clang_command, capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        print(f"Compilation timed out for N={n}!", file=sys.stderr)
        return {}

    if result.returncode != 0:
        print(f"Compilation failed for N={n}!", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return {}

    durations_us = analyze_trace_file(trace_file)
    return {
        key.replace("_us", "_ms"): value / 1000.0 for key, value in durations_us.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate, compile, and benchmark C++ test cases for Clang's lifetime analysis."
    )
    parser.add_argument(
        "--clang-binary", type=str, required=True, help="Path to the Clang executable."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save persistent benchmark files. (Default: ./benchmark_results)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Benchmark files will be saved in: {os.path.abspath(args.output_dir)}\n")

    # Maximize 'n' values while keeping execution time under 10s.
    test_configurations = [
        {
            "name": "cycle",
            "title": "Pointer Cycle in Loop",
            "generator_func": generate_cpp_cycle_test,
            "n_values": [50, 75, 100, 200, 300],
        },
        {
            "name": "merge",
            "title": "CFG Merges",
            "generator_func": generate_cpp_merge_test,
            "n_values": [400, 1000, 2000, 5000],
        },
        {
            "name": "nested_loops",
            "title": "Deeply Nested Loops",
            "generator_func": generate_cpp_nested_loop_test,
            "n_values": [50, 100, 150, 200],
        },
        {
            "name": "switch_fan_out",
            "title": "Switch Fan-out",
            "generator_func": generate_cpp_switch_fan_out_test,
            "n_values": [500, 1000, 2000, 4000],
        },
    ]

    results = {}

    print("Running performance benchmarks...")
    for config in test_configurations:
        test_name = config["name"]
        results[test_name] = {
            "title": config["title"],
            "n": [],
            "lifetime_ms": [],
            "total_ms": [],
            "fact_gen_ms": [],
            "loan_prop_ms": [],
            "live_origins_ms": [],
        }
        for n in config["n_values"]:
            durations_ms = run_single_test(
                args.clang_binary,
                args.output_dir,
                test_name,
                config["generator_func"],
                n,
            )
            if durations_ms:
                results[test_name]["n"].append(n)
                for key, value in durations_ms.items():
                    results[test_name][key].append(value)

                print(
                    f"    Total Analysis: {human_readable_time(durations_ms['lifetime_ms'])} | "
                    f"FactGen: {human_readable_time(durations_ms['fact_gen_ms'])} | "
                    f"LoanProp: {human_readable_time(durations_ms['loan_prop_ms'])} | "
                    f"LiveOrigins: {human_readable_time(durations_ms['live_origins_ms'])}"
                )

    print("\n\n" + "=" * 80)
    print("Generating Markdown Report...")
    print("=" * 80 + "\n")

    markdown_report = generate_markdown_report(results)
    print(markdown_report)

    report_filename = os.path.join(args.output_dir, "performance_report.md")
    with open(report_filename, "w") as f:
        f.write(markdown_report)
    print(f"Report saved to: {report_filename}")
