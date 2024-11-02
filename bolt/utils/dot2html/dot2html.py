#!/usr/bin/env python3
import argparse
import os
import sys

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
HTML_TEMPLATE_NAME = 'd3-graphviz-template.html'
HTML_TEMPLATE_PATH = os.path.join(BASE_PATH, HTML_TEMPLATE_NAME)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dotfile', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='Input .dot file, reads from stdin if not set')
    parser.add_argument('htmlfile', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help='Output .html file, writes to stdout if not set')
    args = parser.parse_args()

    template = open(HTML_TEMPLATE_PATH, 'r')

    for line in template:
        if "<INSERT_DOT>" in line:
            print(args.dotfile.read(), file=args.htmlfile, end='')
        else:
            print(line, file=args.htmlfile, end='')

if __name__ == "__main__":
    main()
