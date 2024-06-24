#!/usr/bin/env python3
import os
import logging
import subprocess

def config_logger():
    LOG_FORMAT = '%(asctime)s %(levelname)s [%(name)s] %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    logging.basicConfig(level=logging.INFO,
                        format=LOG_FORMAT, datefmt=DATE_FORMAT)

config_logger()

REAL_WORLD_DIR = 'vul-parser-benchmark/tests'

def update_paths(root):
    src = '/home/thebesttv/vul/llvm-project/graph-generation'
    dst = root
    logging.info(f'Updating paths: {src} -> {dst}')
    cmd = '''
    sed -i 's|{src}|{dst}|g' $(find {dst} -type f -name '*.json')
'''.format(src=src, dst=dst)
    subprocess.run(cmd, shell=True, check=True)

def get_directories(root):
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

def get_test_cases(root):
    # get all directories under root
    directories = get_directories(root)
    # 额外处理真实项目测试集：https://github.com/thebesttv/vul-parser-benchmark
    if 'vul-parser-benchmark' in directories:
        directories += [os.path.join(REAL_WORLD_DIR, d)
                        for d in get_directories(os.path.join(root, REAL_WORLD_DIR))]
    logging.info(f'Found {len(directories)} directories')
    results = []
    for dir in directories:
        state = ''
        input_path = os.path.join(root, dir, 'input.json')
        output_path = os.path.join(root, dir, 'output.json')
        if not os.path.exists(input_path):
            state = 'missing input.json'
        elif not os.path.exists(output_path):
            state = 'missing output.json'
        else:
            state = 'ok'
            results.append((root, dir, input_path, output_path))
        logging.info(f'  {dir}: {state}')
    logging.info(f'Found {len(results)} test cases')
    return results

def read_output(path):
    with open(path, 'r') as f:
        return f.read().strip()

def run_case(root, dir, input_path, output_path, tool_path):
    logging.info(f"Running test case: {dir}")

    extra_options = []
    if dir.startswith(REAL_WORLD_DIR):
        logging.info("  Real-world testcase! Do not generate NPE good source")
        extra_options += ['--no-npe-good-source', '--no-nodes']

    original_output = read_output(output_path)
    os.remove(output_path)

    cmd = f'{tool_path} {" ".join(extra_options)} {input_path}'
    subprocess.run(cmd, shell=True, \
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, \
                   check=True)

    new_output = read_output(output_path)
    if original_output != new_output:
        os.remove(output_path)
        logging.error(f"  Output mismatch: {dir}")
        logging.info("======== Original ========")
        print(original_output)
        logging.info("======== New ========")
        print(new_output)
        raise Exception(f"Output mismatch: {dir}")

if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.realpath(__file__))
    tool_path = os.path.join(ROOT, 'tool')
    assert os.path.exists(tool_path), f'{tool_path} does not exist'

    logging.info(f'Test directory: {ROOT}')
    logging.info(f'Tool script:    {tool_path}')

    update_paths(ROOT)

    test_cases = get_test_cases(ROOT)
    for test in test_cases:
        run_case(*test, tool_path)
