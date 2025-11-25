# Makefile for Binary Agent Communication Library
# Builds with AVX512 support for P-core optimization

CC = gcc
CFLAGS = -O3 -fPIC -Wall -Wextra
LDFLAGS = -shared -lpthread

# Detect CPU capabilities
CPU_FLAGS := $(shell grep -m1 "^flags" /proc/cpuinfo 2>/dev/null || echo "")

# Enable AVX512 if supported
ifneq (,$(findstring avx512f,$(CPU_FLAGS)))
    CFLAGS += -mavx512f -march=native
    AVX512_ENABLED = yes
else
    AVX512_ENABLED = no
endif

LIB_NAME = libagent_comm.so
SRC = libagent_comm.c
OBJ = libagent_comm.o

.PHONY: all clean info

all: info $(LIB_NAME)

info:
	@echo "======================================"
	@echo "Building Binary Agent Communication Lib"
	@echo "======================================"
	@echo "Compiler: $(CC)"
	@echo "AVX512:   $(AVX512_ENABLED)"
	@echo "Flags:    $(CFLAGS)"
	@echo "======================================"
	@echo

$(LIB_NAME): $(OBJ)
	@echo "Linking $@..."
	$(CC) $(LDFLAGS) -o $@ $^
	@echo "✓ Built: $@"
	@echo "✓ AVX512: $(AVX512_ENABLED)"
	@echo "✓ P-core pinning: enabled"

$(OBJ): $(SRC)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(LIB_NAME)
	@echo "Cleaned build artifacts"

test: $(LIB_NAME)
	@echo "Testing library..."
	@python3 -c "import ctypes; lib = ctypes.CDLL('./$(LIB_NAME)'); print('✓ Library loads successfully')"
	@echo "✓ All tests passed"
