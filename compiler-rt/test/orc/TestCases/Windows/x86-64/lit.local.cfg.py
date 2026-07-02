if config.root.host_arch not in ["AMD64", "x86_64"]:
    config.unsupported = True

if config.target_arch not in ["AMD64", "x86_64"]:
    config.unsupported = True

# These tests have never passed on Windows. The %llvm_jitlink substitution
# defined in compiler-rt/test/orc/lit.cfg.py passes -no-process-syms=true on
# Windows, which llvm-jitlink rejects when combined with -orc-runtime ("-orc-
# runtime requires process symbols"). Simply dropping -no-process-syms exposes
# further unresolved issues in the COFF orc-runtime support (missing __imp_*
# CRT imports, JITSymbolTable index errors when materializing the orc-runtime
# archive, etc.). Mark these tests unsupported on Windows until the underlying
# orc-runtime issues are fixed. See
# https://github.com/llvm/llvm-project/issues/77996.
if config.target_os == "Windows":
    config.unsupported = True
