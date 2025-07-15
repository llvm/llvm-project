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


def analyze_trace_file(trace_path: str) -> tuple[float, float]:
    """
    Parses the -ftime-trace JSON output to find durations.

    Returns:
        A tuple of (lifetime_analysis_duration_us, total_clang_duration_us).
    """
    lifetime_duration = 0.0
    total_duration = 0.0
    try:
        with open(trace_path, "r") as f:
            trace_data = json.load(f)
            for event in trace_data.get("traceEvents", []):
                if event.get("name") == "LifetimeSafetyAnalysis":
                    lifetime_duration += float(event.get("dur", 0))
                if event.get("name") == "ExecuteCompiler":
                    total_duration += float(event.get("dur", 0))

    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing trace file {trace_path}: {e}", file=sys.stderr)
        return 0.0, 0.0
    return lifetime_duration, total_duration


def power_law(n, c, k):
    """Represents the power law function: y = c * n^k"""
    return c * np.power(n, k)


def human_readable_time(ms: float) -> str:
    """Converts milliseconds to a human-readable string (ms or s)."""
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.2f} ms"


def generate_markdown_report(results: dict) -> str:
    """Generates a Markdown-formatted report from the benchmark results."""
    report = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    report.append(f"# Lifetime Analysis Performance Report")
    report.append(f"> Generated on: {timestamp}")
    report.append("\n---\n")

    for test_name, data in results.items():
        title = data["title"]
        report.append(f"## Test Case: {title}")
        report.append("")

        # Table header
        report.append("| N   | Analysis Time | Total Clang Time |")
        report.append("|:----|--------------:|-----------------:|")

        # Table rows
        n_data = np.array(data["n"])
        analysis_data = np.array(data["lifetime_ms"])
        total_data = np.array(data["total_ms"])
        for i in range(len(n_data)):
            analysis_str = human_readable_time(analysis_data[i])
            total_str = human_readable_time(total_data[i])
            report.append(f"| {n_data[i]:<3} | {analysis_str:>13} | {total_str:>16} |")

        report.append("")

        # Complexity analysis
        report.append(f"**Complexity Analysis:**")
        try:
            # Curve fitting requires at least 3 points
            if len(n_data) < 3:
                raise ValueError("Not enough data points to perform curve fitting.")

            popt, pcov = curve_fit(
                power_law, n_data, analysis_data, p0=[0, 2], maxfev=5000
            )
            _, k = popt

            # Confidence Interval for k
            alpha = 0.05  # 95% confidence
            dof = max(0, len(n_data) - len(popt))  # degrees of freedom
            t_val = t.ppf(1.0 - alpha / 2.0, dof)
            # Standard error of the parameters
            perr = np.sqrt(np.diag(pcov))
            k_stderr = perr[1]
            k_ci_lower = k - t_val * k_stderr
            k_ci_upper = k + t_val * k_stderr

            report.append(
                f"- The performance for this case scales approx. as **O(n<sup>{k:.2f}</sup>)**."
            )
            report.append(
                f"- **95% Confidence interval for exponent:** `[{k_ci_lower:.2f}, {k_ci_upper:.2f}]`."
            )

        except (RuntimeError, ValueError) as e:
            report.append(f"- Could not determine a best-fit curve for the data: {e}")

        report.append("\n---\n")

    return "\n".join(report)


def run_single_test(
    clang_binary: str, output_dir: str, test_name: str, generator_func, n: int
) -> tuple[float, float]:
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
        "-Wexperimental-lifetime-safety",
        "-std=c++17",
        source_file,
    ]

    result = subprocess.run(clang_command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Compilation failed for N={n}!", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 0.0, 0.0

    lifetime_us, total_us = analyze_trace_file(trace_file)

    return lifetime_us / 1000.0, total_us / 1000.0


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

    test_configurations = [
        {
            "name": "cycle",
            "title": "Pointer Cycle in Loop",
            "generator_func": generate_cpp_cycle_test,
            "n_values": [10, 25, 50, 75, 100, 150],
        },
        {
            "name": "merge",
            "title": "CFG Merges",
            "generator_func": generate_cpp_merge_test,
            "n_values": [10, 50, 100, 200, 400, 800],
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
        }
        for n in config["n_values"]:
            lifetime_ms, total_ms = run_single_test(
                args.clang_binary,
                args.output_dir,
                test_name,
                config["generator_func"],
                n,
            )
            if total_ms > 0:
                results[test_name]["n"].append(n)
                results[test_name]["lifetime_ms"].append(lifetime_ms)
                results[test_name]["total_ms"].append(total_ms)
                print(
                    f"    Total: {human_readable_time(total_ms)} | Analysis: {human_readable_time(lifetime_ms)}"
                )

    print("\n\n" + "=" * 80)
    print("Generating Markdown Report...")
    print("=" * 80 + "\n")

    markdown_report = generate_markdown_report(results)
    print(markdown_report)
