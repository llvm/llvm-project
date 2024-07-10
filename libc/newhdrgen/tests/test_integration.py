import subprocess
import unittest
from pathlib import Path
import os
import argparse
import sys


class TestHeaderGenIntegration(unittest.TestCase):
    def setUp(self):
        output_dir_env = os.getenv("TEST_OUTPUT_DIR")

        self.output_dir = Path(
            output_dir_env if output_dir_env else "libc/newhdrgen/tests/output"
        )

        self.maxDiff = None

        self.source_dir = Path(__file__).resolve().parent.parent.parent.parent

    def run_script(self, yaml_file, h_def_file, output_dir):
        yaml_file = self.source_dir / yaml_file
        h_def_file = self.source_dir / h_def_file
        result = subprocess.run(
            [
                "python3",
                str(self.source_dir / "libc/newhdrgen/yaml_to_classes.py"),
                str(yaml_file),
                str(h_def_file),
                "--output_dir",
                str(output_dir),
            ],
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
        yaml_file = "libc/newhdrgen/tests/input/test_small.yaml"
        h_def_file = "libc/newhdrgen/tests/input/test_small.h.def"
        expected_output_file = (
            self.source_dir / "libc/newhdrgen/tests/expected_output/test_header.h"
        )
        output_file = self.output_dir / "test_small.h"

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.run_script(yaml_file, h_def_file, self.output_dir)

        self.compare_files(output_file, expected_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TestHeaderGenIntegration arguments")
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for generated headers"
    )
    args, remaining_argv = parser.parse_known_args()

    if args.output_dir:
        os.environ["TEST_OUTPUT_DIR"] = args.output_dir

    sys.argv[1:] = remaining_argv

    unittest.main()
