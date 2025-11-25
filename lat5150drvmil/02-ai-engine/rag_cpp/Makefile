# Makefile for AVX-512 Vector Search
#
# Dell Latitude 5450 MIL-SPEC optimized build
# Requires: g++, cython, numpy, openmp

.PHONY: all build test benchmark clean standalone help

# Python executable
PYTHON := python3

# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -O3 -mavx512f -mavx512dq -march=native -pthread -fopenmp -std=c++17

# Build Cython extension
all: build

# Build the Python extension module
build:
	@echo "Building AVX-512 vector search extension..."
	@echo "Target: Dell Latitude 5450 (P-cores 0-5)"
	$(PYTHON) setup.py build_ext --inplace
	@echo "✓ Build complete"
	@echo ""
	@echo "Import in Python:"
	@echo "  from rag_cpp.vector_search import VectorDatabase"

# Run tests
test: build
	@echo "Running tests..."
	$(PYTHON) -c "from vector_search import check_avx512; assert check_avx512(), 'AVX-512 not supported'"
	@echo "✓ AVX-512 support verified"

# Run benchmark
benchmark: build
	@echo "Running benchmark..."
	$(PYTHON) -c "from vector_search import benchmark; benchmark()"

# Build standalone executable (for testing without Python)
standalone:
	@echo "Building standalone benchmark..."
	$(CXX) $(CXXFLAGS) -DSTANDALONE_BUILD vector_search_avx512.cpp -o vector_search_standalone
	@echo "✓ Standalone build complete"
	@echo ""
	@echo "Run: ./vector_search_standalone"

# Run standalone benchmark
benchmark-standalone: standalone
	@echo "Running standalone benchmark..."
	./vector_search_standalone

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -f vector_search.cpp
	rm -f vector_search*.so
	rm -f vector_search_standalone
	rm -f *.o
	@echo "✓ Clean complete"

# Check CPU features
cpu-info:
	@echo "CPU Features:"
	@lscpu | grep -E "Model name|Flags" | head -n 2
	@echo ""
	@echo "AVX-512 Support:"
	@if lscpu | grep -q avx512f; then \
		echo "  ✓ AVX-512F (Foundation) supported"; \
	else \
		echo "  ✗ AVX-512F NOT supported"; \
	fi
	@if lscpu | grep -q avx512dq; then \
		echo "  ✓ AVX-512DQ (Doubleword/Quadword) supported"; \
	else \
		echo "  ✗ AVX-512DQ NOT supported"; \
	fi

# Display help
help:
	@echo "AVX-512 Vector Search - Build System"
	@echo "===================================="
	@echo ""
	@echo "Targets:"
	@echo "  make build              - Build Python extension (default)"
	@echo "  make test               - Verify AVX-512 support"
	@echo "  make benchmark          - Run Python benchmark"
	@echo "  make standalone         - Build C++ standalone executable"
	@echo "  make benchmark-standalone - Run standalone benchmark"
	@echo "  make clean              - Remove build artifacts"
	@echo "  make cpu-info           - Display CPU features"
	@echo "  make help               - Display this help"
	@echo ""
	@echo "Hardware Target:"
	@echo "  Dell Latitude 5450 MIL-SPEC"
	@echo "  P-cores: 0-5 (AVX-512 enabled)"
	@echo "  E-cores: 6-11 (no AVX-512)"
	@echo ""
	@echo "Requirements:"
	@echo "  - g++ with AVX-512 support"
	@echo "  - Python 3.8+"
	@echo "  - Cython, NumPy"
	@echo "  - OpenMP"
