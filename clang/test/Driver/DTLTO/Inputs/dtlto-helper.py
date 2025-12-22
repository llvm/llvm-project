from pathlib import Path
import sys

# Arg 1: "clang" path.
p = Path(sys.argv[1])
print(f"clang-name:{p.resolve().name}")
# Arg 2: Non-zero for LLVM driver.
if sys.argv[2] != "0":
    print(f'prepend-arg:"--thinlto-remote-compiler-prepend-arg={p.name}"')
else:
    print("prepend-arg: ")
