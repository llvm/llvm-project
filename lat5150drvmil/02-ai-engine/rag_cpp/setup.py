#!/usr/bin/env python3
"""
Build script for AVX-512 optimized vector search

Compiles C++ code with Cython wrapper for Python integration

Usage:
    python setup.py build_ext --inplace

Requirements:
    - g++ with AVX-512 support
    - Cython
    - NumPy
    - OpenMP

Compiler flags:
    - -O3: Maximum optimization
    - -mavx512f: AVX-512 Foundation instructions
    - -mavx512dq: AVX-512 Doubleword and Quadword instructions
    - -march=native: Optimize for current CPU
    - -fopenmp: OpenMP parallel processing
    - -pthread: POSIX threads for core pinning
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import platform

# Determine if we're on Linux (required for P-core pinning)
is_linux = platform.system() == "Linux"

if not is_linux:
    print("⚠️  Warning: P-core pinning only supported on Linux")
    print("   Building without thread affinity features")

# Compiler and linker flags
extra_compile_args = [
    "-O3",                    # Maximum optimization
    "-mavx512f",             # AVX-512 Foundation
    "-mavx512dq",            # AVX-512 Doubleword and Quadword
    "-march=native",         # Optimize for current CPU
    "-fopenmp",              # OpenMP
    "-fPIC",                 # Position independent code
    "-std=c++17",            # C++17 standard
]

extra_link_args = [
    "-fopenmp",              # OpenMP
]

if is_linux:
    extra_link_args.append("-lpthread")  # POSIX threads

# Extension module
extensions = [
    Extension(
        name="vector_search",
        sources=["vector_search.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include(), "."],
    )
]

# Build
setup(
    name="vector_search",
    version="1.0.0",
    description="AVX-512 optimized vector search for Dell Latitude 5450",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,      # Disable bounds checking for speed
            "wraparound": False,        # Disable negative indexing
            "initializedcheck": False,  # Disable memoryview initialization checks
            "cdivision": True,          # Use C division semantics
        }
    ),
    install_requires=[
        "numpy>=1.20.0",
        "cython>=0.29.0",
    ],
    python_requires=">=3.8",
)
