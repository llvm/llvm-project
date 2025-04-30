# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

build:
	@echo ------------------------------------- building test $(TEST)
	$(FC) $(FFLAGS) $(SRC)/$(TEST).f90 -o $(TEST).$(EXE) > $(TEST).rslt 2>&1

run:
	@echo ------------------------------------ executing test $(TEST)
	./$(TEST).$(EXE)

verify: $(TEST).rslt
	@echo ------------------------------------ verifying test $(TEST)
	@cat $(TEST).rslt | grep -v dlopen | grep -v mktemp > tmp || true
	@mv tmp $(TEST).rslt
	@if ! grep -q warn $(TEST).rslt; then \
	  echo "PASS"; \
	else \
	  grep -i warn $(TEST).rslt; \
	  echo "FAIL"; \
	fi
