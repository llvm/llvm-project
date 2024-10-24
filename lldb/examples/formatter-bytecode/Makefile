all: test

.PHONY: test
test:
	python3 compiler.py
	mkdir -p _test
	clang++ -std=c++17 test/MyOptional.cpp -g -o _test/MyOptional
	lldb _test/MyOptional -o "command script import test/formatter.py" -o "b -p here" -o "r" -o "v x" -o "v y" -o q
