# Makefile for TPM2 Acceleration Examples

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11
LDFLAGS =

EXAMPLES = secret_crypto

all: $(EXAMPLES)

secret_crypto: secret_level_crypto_example.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✅ Built: $@"

clean:
	rm -f $(EXAMPLES)
	@echo "✅ Cleaned"

install: all
	@echo "Examples built. Run with:"
	@echo "  sudo ./secret_crypto"

.PHONY: all clean install
