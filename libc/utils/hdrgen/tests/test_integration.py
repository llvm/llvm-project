import argparse
import subprocess
import sys
import unittest
from pathlib import Path


class TestHeaderGenIntegration(unittest.TestCase):
    def setUp(self):
        self.output_dir = TestHeaderGenIntegration.output_dir
        self.source_dir = Path(__file__).parent
        self.main_script = self.source_dir.parent / "main.py"

    def run_script(self, yaml_file, output_file, entry_points):
        command = [
            "python3",
            str(self.main_script),
            str(yaml_file),
            "--output",
            str(output_file),
        ]

        for entry_point in entry_points:
            command.extend(["--entry-point", entry_point])

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        result.check_returncode()

    def compare_files(self, generated_file, expected_file):
        with generated_file.open("r") as gen_file:
            gen_content = gen_file.read()
        with expected_file.open("r") as exp_file:
            exp_content = exp_file.read()

        self.assertEqual(gen_content, exp_content)

    def test_generate_header(self):
        yaml_file = self.source_dir / "input/test_small.yaml"
        expected_output_file = self.source_dir / "expected_output/test_header.h"
        output_file = self.output_dir / "test_small.h"
        entry_points = {"func_b", "func_a", "func_c", "func_d", "func_e"}

        self.run_script(yaml_file, output_file, entry_points)

        self.compare_files(output_file, expected_output_file)


def main():
    parser = argparse.ArgumentParser(description="TestHeaderGenIntegration arguments")
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory for generated headers",
        required=True,
    )
    args, remaining_argv = parser.parse_known_args()

    TestHeaderGenIntegration.output_dir = args.output_dir

    sys.argv[1:] = remaining_argv

    unittest.main()


if __name__ == "__main__":
    main()
