#!/bin/bash

ninja -C build || exit 125

./build/bin/llc '/Users/jonathan_roelofs/Library/Application Support/Radar/Downloads/Problem/113994760/rdar113994760-reduced.bc' -filetype=obj -o out || exit 1
