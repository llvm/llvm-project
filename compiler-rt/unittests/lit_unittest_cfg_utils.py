# Put all 64-bit sanitizer tests in the darwin-64bit-sanitizer parallelism
# group. This will only run three of them concurrently.
def darwin_sanitizer_parallelism_group_func(test):
  return "darwin-64bit-sanitizer" if "x86_64" in test.file_path else ""
