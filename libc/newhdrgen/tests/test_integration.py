import subprocess
import unittest
from pathlib import Path

class TestHeaderGenIntegration(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path('tests/output')
        self.maxDiff = None  

    def run_script(self, yaml_file, h_def_file, output_dir):
        result = subprocess.run([
            'python3', 'yaml_to_classes.py', yaml_file, h_def_file, '--output_dir', str(output_dir)
        ], capture_output=True, text=True)

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        result.check_returncode() 

    def compare_files(self, generated_file, expected_file):
        with generated_file.open('r') as gen_file:
            gen_content = gen_file.read()
        with expected_file.open('r') as exp_file:
            exp_content = exp_file.read()
        
        self.assertEqual(gen_content, exp_content)

    def test_generate_header(self):
        # this is for example, will find a way to test everything at once
        yaml_file = Path('tests/input/test_string.yaml')
        h_def_file = Path('tests/input/string.h.def')
        expected_output_file = Path('tests/expected_output/string.h')
        output_file = self.output_dir / 'string.h'

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.run_script(yaml_file, h_def_file, self.output_dir)

        self.compare_files(output_file, expected_output_file)

if __name__ == '__main__':
    unittest.main()
