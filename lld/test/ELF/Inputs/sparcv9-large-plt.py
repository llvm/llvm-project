import sys

count = int(sys.argv[1])
for i in range(count):
    print(f"call foo{i}")
    print(" nop")
