#!/usr/bin/env python3
#
#===- gen_tests.py - clang-doc test generator ----------------*- python -*--===#
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
"""
clang-doc test generator
==========================

Generates tests for clang-doc given a certain set of flags, a prefix for the
test file, and a given clang-doc binary. Please check emitted tests for
accuracy before using.

To generate all current tests:
- Generate mapper tests:
    python gen_tests.py -flag='--dump-mapper' -flag='--doxygen' -flag='--extra-arg=-fmodules-ts' -prefix mapper -use-check-next

- Generate reducer tests:
    python gen_tests.py -flag='--dump-intermediate' -flag='--doxygen' -flag='--extra-arg=-fmodules-ts' -prefix bc -use-check-next

- Generate yaml tests:
    python gen_tests.py -flag='--format=yaml' -flag='--doxygen' -flag='--extra-arg=-fmodules-ts' -prefix yaml -use-check-next

- Generate public decl tests:
    python gen_tests.py -flag='--format=yaml' -flag='--doxygen' -flag='--public' -flag='--extra-arg=-fmodules-ts' -prefix public -use-check-next

- Generate Markdown tests:
    python gen_tests.py -flag='--format=md' -flag='--doxygen' -flag='--public' -flag='--extra-arg=-fmodules-ts' -prefix md

This script was written on/for Linux, and has not been tested on any other
platform and so it may not work.

"""

import argparse
import glob
import os
import re
import shutil
import subprocess

RUN_CLANG_DOC = """
// RUN: clang-doc {0} -p %t %t/test.cpp -output=%t/docs
"""
RUN = """
// RUN: {0} %t/{1} | FileCheck %s --check-prefix CHECK-{2}
"""

CHECK = '// CHECK-{0}: '

CHECK_NEXT = '// CHECK-{0}-NEXT: '

BITCODE_USR = '<USR abbrevid=4 op0=20 op1={{[0-9]+}} op2={{[0-9]+}} op3={{[0-9]+}} op4={{[0-9]+}} op5={{[0-9]+}} op6={{[0-9]+}} op7={{[0-9]+}} op8={{[0-9]+}} op9={{[0-9]+}} op10={{[0-9]+}} op11={{[0-9]+}} op12={{[0-9]+}} op13={{[0-9]+}} op14={{[0-9]+}} op15={{[0-9]+}} op16={{[0-9]+}} op17={{[0-9]+}} op18={{[0-9]+}} op19={{[0-9]+}} op20={{[0-9]+}}/>'
BITCODE_USR_REGEX = r'<USR abbrevid=4 op0=20 op1=[0-9]+ op2=[0-9]+ op3=[0-9]+ op4=[0-9]+ op5=[0-9]+ op6=[0-9]+ op7=[0-9]+ op8=[0-9]+ op9=[0-9]+ op10=[0-9]+ op11=[0-9]+ op12=[0-9]+ op13=[0-9]+ op14=[0-9]+ op15=[0-9]+ op16=[0-9]+ op17=[0-9]+ op18=[0-9]+ op19=[0-9]+ op20=[0-9]+/>'
YAML_USR = "USR:             '{{[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]}}'"
YAML_USR_REGEX = r"USR:             '[0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z][0-9A-Z]'"

def clear_test_prefix_files(prefix, tests_path):
    if os.path.isdir(tests_path):
        for root, dirs, files in os.walk(tests_path):
            for filename in files:
                if filename.startswith(prefix):
                    os.remove(os.path.join(root, filename))


def copy_to_test_file(test_case_path, test_cases_path):
    # Copy file to 'test.cpp' to preserve file-dependent USRs
    test_file = os.path.join(test_cases_path, 'test.cpp')
    shutil.copyfile(test_case_path, test_file)
    return test_file


def run_clang_doc(args, out_dir, test_file):
    # Run clang-doc.
    current_cmd = [args.clangdoc]
    current_cmd.extend(args.flags)
    current_cmd.append('--output=' + out_dir)
    current_cmd.append(test_file)
    print('Running ' + ' '.join(current_cmd))
    return_code = subprocess.call(current_cmd)
    if return_code:
        return 1
    return 0


def get_test_case_code(test_case_path, flags):
    # Get the test case code
    code = ''
    with open(test_case_path, 'r') as code_file:
        code = code_file.read()

    code += RUN_CLANG_DOC.format(flags)
    return code


def get_output(root, out_file, case_out_path, flags, checkname, bcanalyzer,
                check_next=True):
    output = ''
    run_cmd = ''
    if '--dump-mapper' in flags or '--dump-intermediate' in flags:
        # Run llvm-bcanalyzer
        output = subprocess.check_output(
            [bcanalyzer, '--dump',
             os.path.join(root, out_file)])
        output = output[:output.find('Summary of ')].rstrip()
        run_cmd = RUN.format('llvm-bcanalyzer --dump',
                             os.path.join('docs', 'bc', out_file), checkname)
    else:
        # Run cat
        output = subprocess.check_output(['cat', os.path.join(root, out_file)])
        run_cmd = RUN.format(
            'cat',
            os.path.join('docs', os.path.relpath(root, case_out_path),
                         out_file), checkname)

    # Format output.
    output = output.replace('blob data = \'test\'', 'blob data = \'{{.*}}\'')
    output = re.sub(YAML_USR_REGEX, YAML_USR, output)
    output = re.sub(BITCODE_USR_REGEX, BITCODE_USR, output)
    output = CHECK.format(checkname) + output.rstrip()
    
    if check_next:
      check_comment = CHECK_NEXT.format(checkname)
    else:
      check_comment = CHECK.format(checkname)
    
    output = output.replace('\n', '\n' + check_comment)
    output = run_cmd + output.replace('%s\n' % check_comment, "")

    return output + '\n'


def main():
    parser = argparse.ArgumentParser(description='Generate clang-doc tests.')
    parser.add_argument(
        '-flag',
        action='append',
        default=[],
        dest='flags',
        help='Flags to pass to clang-doc.')
    parser.add_argument(
        '-prefix',
        type=str,
        default='',
        dest='prefix',
        help='Prefix for this test group.')
    parser.add_argument(
        '-clang-doc-binary',
        dest='clangdoc',
        metavar="PATH",
        default='clang-doc',
        help='path to clang-doc binary')
    parser.add_argument(
        '-llvm-bcanalyzer-binary',
        dest='bcanalyzer',
        metavar="PATH",
        default='llvm-bcanalyzer',
        help='path to llvm-bcanalyzer binary')
    parser.add_argument(
        '-use-check-next',
        dest='check_next',
        default=False,
        action='store_true',
        help='Whether or not to use CHECK-NEXT in the resulting tests.')
    args = parser.parse_args()

    flags = ' '.join(args.flags)

    clang_doc_path = os.path.dirname(__file__)
    tests_path = os.path.join(clang_doc_path, '..', 'test', 'clang-doc')
    test_cases_path = os.path.join(tests_path, 'test_cases')

    clear_test_prefix_files(args.prefix, tests_path)

    for test_case_path in glob.glob(os.path.join(test_cases_path, '*')):
        if test_case_path.endswith(
                'compile_flags.txt') or test_case_path.endswith(
                    'compile_commands.json'):
            continue

        # Name of this test case
        case_name = os.path.basename(test_case_path).split('.')[0]

        test_file = copy_to_test_file(test_case_path, test_cases_path)
        out_dir = os.path.join(test_cases_path, case_name)

        if run_clang_doc(args, out_dir, test_file):
            return 1

        # Retrieve output and format as FileCheck tests
        all_output = ''
        num_outputs = 0
        for root, dirs, files in os.walk(out_dir):
            for out_file in files:
                # Make the file check the first 3 letters (there's a very small chance
                # that this will collide, but the fix is to simply change the decl name)
                usr = os.path.basename(out_file).split('.')
                # If the usr is less than 2, this isn't one of the test files.
                if len(usr) < 2:
                    continue
                all_output += get_output(root, out_file, out_dir, args.flags,
                                         num_outputs, args.bcanalyzer, 
                                         args.check_next)
                num_outputs += 1

        # Add test case code to test
        all_output = get_test_case_code(test_case_path,
                                        flags) + '\n' + all_output

        # Write to test case file in /test.
        test_out_path = os.path.join(
            tests_path, args.prefix + '-' + os.path.basename(test_case_path))
        with open(test_out_path, 'w+') as o:
            o.write(all_output)

        # Clean up
        shutil.rmtree(out_dir)
        os.remove(test_file)


if __name__ == '__main__':
    main()
