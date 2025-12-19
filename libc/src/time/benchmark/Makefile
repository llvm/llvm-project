# Simple Makefile for time conversion benchmark
# Usage: make run

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra
INCLUDES = -I.. -I../../.. -I../../../include -I../../../hdr

# Define namespace macros
DEFINES = -DLIBC_NAMESPACE=__llvm_libc -DLIBC_NAMESPACE_DECL=__llvm_libc

# Source files
SOURCES = benchmark_time_conversion.cpp ../time_utils.cpp ../fast_date.cpp
TARGET = benchmark_time_conversion

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) $(SOURCES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
